"""The MAST judge: windows a trace in code and asks Jev about each window."""

import contextvars
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import dspy

from mast_judge import replay
from mast_judge.ladder import LADDER_MODES, RUNGS, JudgeLadder, combine, soft_scores
from mast_judge.signatures import JudgeOutcome, JudgeWindow
from mast_judge.taxonomy import MODES

# ChatDev prints this line between most messages. It carries no signal.
NOISE = re.compile(r"^\[[^\]]*\] flask app\.py did not start for online log\n?", re.MULTILINE)


def clean(trace: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", NOISE.sub("", trace))


def split_windows(text: str, size: int) -> list[str]:
    """Split on line boundaries into windows of at most `size` characters."""
    windows, current, length = [], [], 0
    for line in text.splitlines(keepends=True):
        while len(line) > size:
            if current:
                windows.append("".join(current))
                current, length = [], 0
            windows.append(line[:size])
            line = line[size:]
        if length + len(line) > size and current:
            windows.append("".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line)
    if current:
        windows.append("".join(current))
    return windows or [""]


@lru_cache(maxsize=4096)
def prepare(trace: str, window_chars: int, task_chars: int) -> tuple[str, tuple[str, ...]]:
    """The task excerpt and the windows of a trace. Cached because calibration reruns every trace."""
    text = clean(trace)
    return text[:task_chars], tuple(split_windows(text, window_chars))


def position(index: int, count: int) -> str:
    if count == 1:
        return "The excerpt is the entire log."
    if index == 0:
        return "The excerpt is the opening of the log. More of the log follows."
    if index == count - 1:
        return "The excerpt is the end of the log. Earlier parts of the log came before it."
    return "The excerpt is from the middle of the log. Parts of the log come before and after it."


class MASTJudge(dspy.Module):
    """Label a multi-agent trace with the 14 MAST failure modes.

    Every window gets one Noul per mode, and a mode is present when any window
    passes its threshold. With `use_outcome`, the end of the log also gets a
    Choice about how the run concluded and a Score for how thoroughly the
    agents verified their result. Those answers can add 1.5, 3.1, 3.2, and 3.3.
    With `use_ladder`, every window also gets the ladder rungs, and the rungs
    decide the modes in `LADDER_MODES` in place of their single Nouls.
    """

    def __init__(self, window_chars=40_000, task_chars=3_000, use_outcome=False, use_ladder=False, max_workers=4):
        super().__init__()
        self.window_chars = window_chars
        self.task_chars = task_chars
        self.use_outcome = use_outcome
        self.use_ladder = use_ladder
        self.max_workers = max_workers
        self.judge_window = dspy.Predict(JudgeWindow)
        if use_outcome:
            self.judge_outcome = dspy.Predict(JudgeOutcome)
        if use_ladder:
            self.judge_ladder = dspy.Predict(JudgeLadder)

    def forward(self, trace: str):
        task, windows = prepare(trace, self.window_chars, self.task_chars)
        inputs = [{"task": task, "window_position": position(i, len(windows)), "trace_window": w} for i, w in enumerate(windows)]
        calls = [(self.judge_window, x) for x in inputs]
        if self.use_ladder:
            calls += [(self.judge_ladder, x) for x in inputs]
        if self.use_outcome:
            calls.append((self.judge_outcome, {"task": task, "log_ending": windows[-1]}))
        results = self._run(calls)

        window_results = results[: len(windows)]
        labels = {m.code: any(bool(r[m.field]) for r in window_results) for m in MODES}
        probabilities = {m.code: max(r[m.field].probability for r in window_results) for m in MODES}
        if self.use_ladder:
            ladder_results = results[len(windows) : 2 * len(windows)]
            decided = combine({name: any(bool(r[name]) for r in ladder_results) for name in RUNGS})
            # The soft score ranks traces by the same logic as the decision.
            soft = soft_scores({name: max(r[name].probability for r in ladder_results) for name in RUNGS})
            for code in LADDER_MODES:
                labels[code], probabilities[code] = decided[code], soft[code]
        outcome = results[-1] if self.use_outcome else None
        if outcome is not None:
            labels["1.5"] |= outcome.ending.value == "ran_on"
            labels["3.1"] |= outcome.ending.value == "stopped_early"
            labels["3.2"] |= outcome.verification.level <= 1
            labels["3.3"] |= outcome.verification.level == 2

        return dspy.Prediction(
            mode_labels=labels,
            failure_modes=[code for code, present in labels.items() if present],
            probabilities=probabilities,
            ending=outcome.ending if outcome else None,
            verification=outcome.verification if outcome else None,
            windows=len(windows),
        )

    def _run(self, calls):
        # Replayed answers need no network, so threads would only add overhead.
        if replay.active() or len(calls) == 1:
            return [predict(**kwargs) for predict, kwargs in calls]
        # Copying the context keeps DSPy settings, including ReAnchor's evidence log, in each worker.
        with ThreadPoolExecutor(self.max_workers) as pool:
            futures = [pool.submit(contextvars.copy_context().run, predict, **kwargs) for predict, kwargs in calls]
            return [f.result() for f in futures]
