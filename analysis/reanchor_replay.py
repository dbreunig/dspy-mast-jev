"""Check that calibrating with `replay()` reproduces an earlier ReAnchor run, and time it.

    uv run python analysis/reanchor_replay.py runs/full/nouls/report.json
"""

import json
import sys
import time

import dspy
from dotenv import load_dotenv
from dspy.experimental import ReAnchor, TypeSafe

from mast_judge.data import load_examples, split
from mast_judge.judge import MASTJudge
from mast_judge.metrics import trace_f1
from mast_judge.replay import replay


def main(reference_path="runs/full/nouls/report.json"):
    load_dotenv(".env")
    dspy.configure(lm=TypeSafe(model="jev-1.13.0"))
    trainset, _, _ = split(load_examples())
    optimizer = ReAnchor(metric=trace_f1, num_threads=8)
    start = time.time()
    with replay():
        optimizer.compile(MASTJudge(use_outcome=False), trainset=trainset)
    minutes = (time.time() - start) / 60

    with open(reference_path) as f:
        reference = json.load(f)
    fitted = {row["field"]: row.get("value") for row in optimizer.report["fitted"]}
    expected = {row["field"]: row.get("value") for row in reference["fitted"]}
    print(f"ReAnchor with replay: {minutes:.1f} minutes")
    print(f"train score {optimizer.report['train_score_before']} -> {optimizer.report['train_score']} "
          f"(reference {reference['train_score_before']} -> {reference['train_score']})")
    print("fitted settings identical to the reference:", fitted == expected)
    for field in fitted:
        if fitted[field] != expected[field]:
            print(f"  {field}: replay={fitted[field]} reference={expected[field]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
