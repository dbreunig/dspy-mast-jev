"""Calibrate the MAST judge with ReAnchor and evaluate it on held-out traces.

Compiles each variant named in --variants (by default, window Nouls alone and window
Nouls with the ladder questions). Picks the variant that scores higher on dev, then
scores the uncalibrated and calibrated versions on the test split. With --metric
balanced, dev selection uses macro F1.

    uv run python scripts/optimize.py --metric balanced --out runs/balanced
"""

import argparse
import json
import logging
from pathlib import Path

import dspy
from dotenv import load_dotenv
from dspy.experimental import ReAnchor, TypeSafe

from mast_judge.data import load_examples, split
from mast_judge.judge import MASTJudge
from mast_judge.replay import replay
from mast_judge.metrics import balanced_metric, format_report, report, trace_f1


VARIANTS = {
    "nouls": {},
    "nouls+outcome": {"use_outcome": True},
    "nouls+ladder": {"use_ladder": True},
}


def evaluate(program, examples, threads, label):
    result = dspy.Evaluate(devset=examples, metric=trace_f1, num_threads=threads, display_progress=True)(program)
    preds = [pred for _, pred, _ in result.results]
    rep = report(examples, preds)
    print(f"\n== {label}\n{format_report(rep)}", flush=True)
    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-chars", type=int, default=40_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default="runs/latest")
    parser.add_argument(
        "--metric",
        choices=["trace_f1", "balanced"],
        default="trace_f1",
        help="trace_f1 favors per-trace agreement. balanced weighs every mode equally, so rare modes stay on.",
    )
    parser.add_argument("--variants", default="nouls,nouls+ladder", help=f"Comma-separated, from {sorted(VARIANTS)}")
    parser.add_argument("--limit", type=int, help="Use at most this many examples from each split")
    args = parser.parse_args()

    load_dotenv(".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for name in ("httpx", "httpx2", "typesafe_sdk"):
        logging.getLogger(name).setLevel(logging.WARNING)
    dspy.configure(lm=TypeSafe(model="jev-1.13.0"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trainset, devset, testset = split(load_examples())
    if args.limit:
        trainset, devset, testset = (s[: args.limit] for s in (trainset, devset, testset))
    print(f"train={len(trainset)} dev={len(devset)} test={len(testset)}", flush=True)

    metric = trace_f1 if args.metric == "trace_f1" else balanced_metric(trainset)
    selection = "trace_f1" if args.metric == "trace_f1" else "macro_f1"

    results, compiled = {}, {}
    for name in args.variants.split(","):
        judge = MASTJudge(window_chars=args.window_chars, **VARIANTS[name])
        results[f"{name}/dev/base"] = evaluate(judge, devset, args.threads, f"{name}: dev, uncalibrated")
        optimizer = ReAnchor(metric=metric, num_threads=args.threads, log_dir=out / name)
        with replay():
            compiled[name] = optimizer.compile(judge, trainset=trainset, valset=devset)
        results[f"{name}/fitted"] = optimizer.report
        results[f"{name}/dev/calibrated"] = evaluate(compiled[name], devset, args.threads, f"{name}: dev, calibrated")
        compiled[name].save(out / f"{name.replace('+', '_')}.json")

    best = max(compiled, key=lambda n: results[f"{n}/dev/calibrated"][selection])
    print(f"\nSelected on dev: {best}", flush=True)
    base = MASTJudge(window_chars=args.window_chars, **VARIANTS[best])
    results["selected"] = best
    results["test/base"] = evaluate(base, testset, args.threads, f"{best}: test, uncalibrated")
    results["test/calibrated"] = evaluate(compiled[best], testset, args.threads, f"{best}: test, calibrated")
    compiled[best].save(out / "mast_judge.json")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
