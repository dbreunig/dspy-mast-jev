"""DSPy signatures for judging multi-agent traces with Jev.

Every output is a decision type, so `dspy.Predict` sends each signature to
Jev as one System One request: the inputs become `state` and each output
becomes a question answered in parallel.
"""

import dspy
from dspy.experimental import Choice, Noul, Score

from mast_judge.taxonomy import MODES

TASK_DESC = "The opening of the log, where the task and the agents' roles are stated"

WINDOW_INSTRUCTIONS = (
    "Inspect an excerpt from the log of a multi-agent system working on a task. "
    "Judge only what the excerpt shows. The task statement comes from the start of the same log."
)


def _window_signature():
    fields = {
        "task": (str, dspy.InputField(desc=TASK_DESC)),
        "window_position": (str, dspy.InputField(desc="Where the excerpt sits in the full log")),
        "trace_window": (str, dspy.InputField(desc="The excerpt of the log to judge")),
    }
    for mode in MODES:
        kind = Noul[(True, mode.true), (False, mode.false)]
        fields[mode.field] = (kind, dspy.OutputField(desc=mode.question))
    return dspy.make_signature(fields, WINDOW_INSTRUCTIONS, signature_name="JudgeWindow")


# One Noul per MAST failure mode, asked of every window of a trace.
JudgeWindow = _window_signature()

ENDING = Choice[
    (
        "finished",
        {
            "what": "The agents deliver a result that covers the task, with nothing the task needs left missing, and then stop",
            "examples": ["The coder submits the program, the tester runs it, and the run ends with the passing output"],
        },
    ),
    (
        "stopped_early",
        {
            "what": "The agents stop, or declare the task done or failed, while work, information, or checks the task needs are still missing",
            "examples": [
                "The agent cannot find an access token and marks the task failed without asking for it",
                "The run ends right after the tests are written and before they run",
            ],
        },
    ),
    (
        "ran_on",
        {
            "what": "The agents keep going after they should stop: they repeat the same exchange, keep prompting after the task is done or blocked, or the run ends only at a turn or round limit",
            "examples": [
                "One agent says the problem cannot be solved and the other replies 'Continue' again and again",
                "The log ends with a message that the maximum number of rounds was reached",
            ],
        },
    ),
    (
        "unclear",
        {
            "what": "The log ends without showing how the run concluded, such as a log that stops in the middle of a step",
        },
    ),
]

VERIFICATION = Score[
    "No agent checks the result before the run ends",
    "A check of the result is planned or started, but it is skipped, never run, or left unfinished",
    "A check runs, but it looks only at syntax, format, or whether the program starts, and not at whether the result meets the task",
    "The agents test the result against the task requirements and act on what the test shows",
]


class JudgeOutcome(dspy.Signature):
    """Inspect the end of the log of a multi-agent system working on a task. The task statement comes from the start of the same log."""

    task: str = dspy.InputField(desc=TASK_DESC)
    log_ending: str = dspy.InputField(desc="The final part of the log, up to the last line")
    ending: ENDING = dspy.OutputField(desc="How does the run shown in `inputs.log_ending` conclude?")
    verification: VERIFICATION = dspy.OutputField(
        desc="How thoroughly do the agents in `inputs.log_ending` check their final result against the task in `inputs.task`?"
    )
