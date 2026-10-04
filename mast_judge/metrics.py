"""Metrics for multi-label MAST judgments."""

from mast_judge.taxonomy import MODES


def trace_f1(example, pred, trace=None) -> float:
    """F1 between gold and predicted failure modes for one trace. Two empty sets score 1."""
    gold = {c for c, v in example.mode_labels.items() if v}
    predicted = {c for c, v in pred.mode_labels.items() if v}
    if not gold and not predicted:
        return 1.0
    return 2 * len(gold & predicted) / (len(gold) + len(predicted))


def balanced_metric(examples):
    """Build a per-trace metric whose mean over `examples` is macro balanced accuracy.

    Each mode counts a hit on a positive trace by 1 / (2 * prevalence) and a hit on a
    negative trace by 1 / (2 * (1 - prevalence)). Rare modes then weigh as much as common ones.
    """
    n = len(examples)
    prevalence = {m.code: sum(ex.mode_labels[m.code] for ex in examples) / n for m in MODES}

    def metric(example, pred, trace=None) -> float:
        total = 0.0
        for code, p in prevalence.items():
            gold, guess = example.mode_labels[code], pred.mode_labels[code]
            if 0 < p < 1 and gold == guess:
                total += 1 / (2 * p) if gold else 1 / (2 * (1 - p))
        return total / len(prevalence)

    return metric


def report(examples, preds) -> dict:
    """Per-mode precision, recall, F1, and accuracy, with macro and micro summaries."""
    rows, tp_all, fp_all, fn_all = {}, 0, 0, 0
    for mode in MODES:
        gold = [ex.mode_labels[mode.code] for ex in examples]
        pred = [p.mode_labels[mode.code] for p in preds]
        tp = sum(g and p for g, p in zip(gold, pred))
        fp = sum(p and not g for g, p in zip(gold, pred))
        fn = sum(g and not p for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows[mode.code] = {
            "name": mode.name,
            "support": sum(gold),
            "predicted": sum(pred),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        }
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
    return {
        "modes": rows,
        "macro_f1": sum(r["f1"] for r in rows.values()) / len(rows),
        "micro_f1": 2 * tp_all / (2 * tp_all + fp_all + fn_all) if tp_all else 0.0,
        "mean_accuracy": sum(r["accuracy"] for r in rows.values()) / len(rows),
        "trace_f1": sum(trace_f1(ex, p) for ex, p in zip(examples, preds)) / len(examples),
        "n": len(examples),
    }


def format_report(rep: dict) -> str:
    lines = [f"{'mode':<5} {'name':<34} {'sup':>4} {'pred':>4} {'P':>5} {'R':>5} {'F1':>5} {'acc':>5}"]
    for code, r in rep["modes"].items():
        lines.append(
            f"{code:<5} {r['name']:<34} {r['support']:>4} {r['predicted']:>4} "
            f"{r['precision']:>5.2f} {r['recall']:>5.2f} {r['f1']:>5.2f} {r['accuracy']:>5.2f}"
        )
    lines.append(
        f"n={rep['n']}  trace F1={rep['trace_f1']:.3f}  macro F1={rep['macro_f1']:.3f}  "
        f"micro F1={rep['micro_f1']:.3f}  mean accuracy={rep['mean_accuracy']:.3f}"
    )
    return "\n".join(lines)
