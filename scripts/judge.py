"""Judge one multi-agent trace with a calibrated MAST judge.

    uv run python scripts/judge.py path/to/trace.txt
    uv run python scripts/judge.py path/to/trace.txt --program runs/full/mast_judge.json
"""

import argparse
import json

import dspy
from dotenv import load_dotenv
from dspy.experimental import TypeSafe

from mast_judge.judge import MASTJudge
from mast_judge.taxonomy import CATEGORIES, MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="A text file holding the trace")
    parser.add_argument("--program", default="runs/full/mast_judge.json")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    args = parser.parse_args()

    load_dotenv(".env")
    dspy.configure(lm=TypeSafe(model="jev-1.13.0"))
    with open(args.program) as f:
        saved = json.load(f)
    judge = MASTJudge(use_outcome="judge_outcome" in saved, use_ladder="judge_ladder" in saved)
    judge.load(args.program)

    with open(args.trace) as f:
        pred = judge(trace=f.read())

    if args.json:
        print(json.dumps({"failure_modes": pred.failure_modes, "probabilities": pred.probabilities}, indent=2))
        return
    print(f"Judged {pred.windows} window(s).")
    for mode in MODES:
        mark = "x" if pred.mode_labels[mode.code] else " "
        print(f"[{mark}] {mode.code} {mode.name:<34} p={pred.probabilities[mode.code]:.2f}  ({CATEGORIES[mode.category]})")
    if pred.ending is not None:
        print(f"Ending: {pred.ending.value} (confidence {pred.ending.confidence:.2f})")
        print(f"Verification level: {pred.verification.level} (score {pred.verification.value:.2f})")


if __name__ == "__main__":
    main()
