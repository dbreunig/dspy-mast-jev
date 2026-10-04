"""Fit one threshold per mode to maximize that mode's F1 on train, then score it on test."""

import json

import dspy
from dotenv import load_dotenv
from dspy.experimental import TypeSafe

from mast_judge.data import load_examples, split
from mast_judge.judge import MASTJudge
from mast_judge.taxonomy import MODES


def probabilities(examples):
    result = dspy.Evaluate(devset=examples, metric=lambda e, p, t=None: 0.0, num_threads=8)(MASTJudge(use_outcome=False))
    return [(ex.mode_labels, pred.probabilities) for ex, pred, _ in result.results]


def f1(rows, code, t):
    tp = sum(p[code] >= t and y[code] for y, p in rows)
    fp = sum(p[code] >= t and not y[code] for y, p in rows)
    fn = sum(p[code] < t and y[code] for y, p in rows)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    load_dotenv(".env")
    dspy.configure(lm=TypeSafe(model="jev-1.13.0"))
    trainset, _, testset = split(load_examples())
    train, test = probabilities(trainset), probabilities(testset)
    with open("runs/full/mast_judge.json") as f:
        fields = json.load(f)["judge_window"]["fields"]
    grid = [i / 100 for i in range(1, 100)]
    print(f"{'mode':<5} {'name':<34} {'ReAnchor t':>10} {'test F1':>7} {'own t':>6} {'test F1':>7} {'best F1 on test':>15}")
    totals = [0.0, 0.0, 0.0]
    for m in MODES:
        t_re = fields.get(m.field, {}).get("threshold", 0.5)
        t_own = max(grid, key=lambda t: f1(train, m.code, t))
        best = max(f1(test, m.code, t) for t in grid)
        row = (f1(test, m.code, t_re), f1(test, m.code, t_own), best)
        totals = [a + b for a, b in zip(totals, row)]
        print(f"{m.code:<5} {m.name:<34} {t_re:>10.2f} {row[0]:>7.2f} {t_own:>6.2f} {row[1]:>7.2f} {row[2]:>15.2f}")
    n = len(MODES)
    print(f"{'':<5} {'macro F1':<34} {'':>10} {totals[0] / n:>7.3f} {'':>6} {totals[1] / n:>7.3f} {totals[2] / n:>15.3f}")


if __name__ == "__main__":
    main()
