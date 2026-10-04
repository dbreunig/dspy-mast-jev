"""Measure whether each Noul's probability separates labeled traces, and whether confidence tracks correctness.

Uses cached Jev answers from a previous optimization run. A trace's probability for a
mode is the highest probability across its windows.
"""

import json
import sys

import dspy
from dotenv import load_dotenv
from dspy.experimental import TypeSafe

from mast_judge.data import load_examples, split
from mast_judge.judge import MASTJudge
from mast_judge.taxonomy import MODES


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(program="runs/full/mast_judge.json", out="analysis/test_probabilities.json"):
    load_dotenv(".env")
    dspy.configure(lm=TypeSafe(model="jev-1.13.0"))
    _, _, testset = split(load_examples())
    judge = MASTJudge(use_outcome=False)
    result = dspy.Evaluate(devset=testset, metric=lambda e, p, t=None: 0.0, num_threads=8)(judge)
    rows = [
        {"key": ex.key, "group": ex.group, "labels": ex.mode_labels, "probabilities": pred.probabilities}
        for ex, pred, _ in result.results
    ]
    with open(out, "w") as f:
        json.dump(rows, f)

    with open(program) as f:
        fields = json.load(f)["judge_window"]["fields"]
    thresholds = {m.code: fields.get(m.field, {}).get("threshold", 0.5) for m in MODES}

    print(f"{'mode':<5} {'name':<34} {'pos':>4} {'AUROC':>6}   accuracy by confidence (distance from threshold), n in parentheses")
    bins = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)]
    print(" " * 53 + "   ".join(f"{lo:.2f} to {hi if hi < 1 else 1:.2f}" for lo, hi in bins))
    for m in MODES:
        p = [r["probabilities"][m.code] for r in rows]
        y = [r["labels"][m.code] for r in rows]
        t = thresholds[m.code]
        cells = []
        for lo, hi in bins:
            hits = [(pi >= t) == yi for pi, yi in zip(p, y) if lo <= abs(pi - t) / max(t, 1 - t) < hi]
            cells.append(f"{sum(hits) / len(hits):.2f} ({len(hits):>3})" if hits else "  -   (  0)")
        print(f"{m.code:<5} {m.name:<34} {sum(y):>4} {auroc(p, y):>6.3f}   " + "   ".join(cells))


if __name__ == "__main__":
    main(*sys.argv[1:])
