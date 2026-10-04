"""Compare ladder questions against the original Nouls for modes 1.2, 3.2, and 3.3.

For each trace, a rung's probability is its highest probability across windows.
The script reports:
- each rung's AUROC against its target mode on train,
- AUROC of the original Noul and of the ladder's soft score on test,
- test F1 with thresholds fit on train to maximize that mode's F1, for both.

    uv run python analysis/ladder_experiment.py
"""

import json
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.experimental import TypeSafe

from mast_judge.data import load_examples, split
from mast_judge.judge import MASTJudge, position, prepare
from mast_judge.ladder import RUNGS, JudgeLadder, combine, soft_scores

TARGETS = {
    "1.2": ["roles_defined", "does_other_job", "refuses_own_job", "speaks_as_other"],
    "3.2": ["unchecked_result", "check_skipped", "check_present", "check_executed"],
    "3.3": ["wrong_pass", "check_surface", "check_requirements", "check_executed"],
}
# Rungs that combine() reads for each mode. check_executed is measured but not combined.
COMBINED = {
    "1.2": ["roles_defined", "does_other_job", "refuses_own_job", "speaks_as_other"],
    "3.2": ["unchecked_result", "check_skipped", "check_present"],
    "3.3": ["wrong_pass", "check_surface", "check_requirements"],
}
OUT = Path("analysis/ladder_probabilities.json")


class LadderProbe(MASTJudge):
    """Ask the ladder rungs of every window and keep each rung's highest probability."""

    def __init__(self, **kwargs):
        super().__init__(use_outcome=False, **kwargs)
        self.judge_ladder = dspy.Predict(JudgeLadder)

    def forward(self, trace):
        task, windows = prepare(trace, self.window_chars, self.task_chars)
        inputs = [{"task": task, "window_position": position(i, len(windows)), "trace_window": w} for i, w in enumerate(windows)]
        results = self._run([(self.judge_window, x) for x in inputs] + [(self.judge_ladder, x) for x in inputs])
        original, ladder = results[: len(windows)], results[len(windows) :]
        return dspy.Prediction(
            original={code: max(r[f].probability for r in original) for code, f in [("1.2", "disobey_role_spec"), ("3.2", "no_or_incomplete_verification"), ("3.3", "incorrect_verification")]},
            rungs={name: max(r[name].probability for r in ladder) for name in RUNGS},
        )


def collect():
    if OUT.exists():
        return json.loads(OUT.read_text())
    trainset, devset, testset = split(load_examples())
    data = {}
    for name, examples in [("train", trainset), ("dev", devset), ("test", testset)]:
        result = dspy.Evaluate(devset=examples, metric=lambda e, p, t=None: 0.0, num_threads=8, display_progress=True)(LadderProbe())
        data[name] = [
            {"key": ex.key, "labels": {c: ex.mode_labels[c] for c in TARGETS}, "original": pred.original, "rungs": pred.rungs}
            for ex, pred, _ in result.results
        ]
    OUT.write_text(json.dumps(data))
    return data


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def f1(pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


GRID = [i / 40 for i in range(1, 40)]


def ladder_decisions(rows, code, thresholds):
    return [combine({n: r["rungs"][n] >= thresholds.get(n, 0.5) for n in RUNGS})[code] for r in rows]


def fit_ladder(rows, code):
    """Coordinate ascent on the thresholds of the rungs a mode combines, for that mode's F1."""
    gold = [r["labels"][code] for r in rows]
    thresholds = {n: 0.5 for n in COMBINED[code]}
    for _ in range(4):
        for name in COMBINED[code]:
            thresholds[name] = max(GRID, key=lambda t: f1(ladder_decisions(rows, code, {**thresholds, name: t}), gold))
    return thresholds


def main():
    load_dotenv(".env")
    dspy.configure(lm=TypeSafe(model="jev-1.13.0"))
    data = collect()
    train, test = data["train"] + data["dev"], data["test"]

    print("Rung AUROC against the target mode (train and dev)")
    for code, names in TARGETS.items():
        gold = [r["labels"][code] for r in train]
        cells = [f"{n}={auroc([r['rungs'][n] for r in train], gold):.3f}" for n in names]
        print(f"  {code}: original={auroc([r['original'][code] for r in train], gold):.3f}  " + "  ".join(cells))

    print("\nTest results (thresholds fit on train and dev)")
    print(f"  {'mode':<5} {'AUROC orig':>10} {'AUROC ladder':>12} {'F1 orig':>8} {'F1 ladder':>9}  ladder thresholds")
    for code in TARGETS:
        gold_train = [r["labels"][code] for r in train]
        gold_test = [r["labels"][code] for r in test]
        t_orig = max(GRID, key=lambda t: f1([r["original"][code] >= t for r in train], gold_train))
        thresholds = fit_ladder(train, code)
        print(
            f"  {code:<5} {auroc([r['original'][code] for r in test], gold_test):>10.3f}"
            f" {auroc([soft_scores(r['rungs'])[code] for r in test], gold_test):>12.3f}"
            f" {f1([r['original'][code] >= t_orig for r in test], gold_test):>8.3f}"
            f" {f1(ladder_decisions(test, code, thresholds), gold_test):>9.3f}  {thresholds}"
        )


if __name__ == "__main__":
    main()
