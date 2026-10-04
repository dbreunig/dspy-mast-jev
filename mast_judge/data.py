"""Load the annotated MAST traces as DSPy examples."""

import json
import random
from collections import defaultdict

import dspy
from huggingface_hub import hf_hub_download

from mast_judge.taxonomy import MODE_CODES

REPO_ID = "mcemri/MAST-Data"


def load_examples() -> list[dspy.Example]:
    """Every trace in MAD_full_dataset.json. Missing annotations count as absent."""
    path = hf_hub_download(REPO_ID, "MAD_full_dataset.json", repo_type="dataset")
    with open(path) as f:
        records = json.load(f)
    return [
        dspy.Example(
            trace=r["trace"]["trajectory"],
            mode_labels={code: bool(r["mast_annotation"].get(code)) for code in MODE_CODES},
            key=f'{r["trace"]["key"]}/{r["trace_id"]}',
            group=f'{r["mas_name"]}/{r["benchmark_name"]}/{r["llm_name"]}',
        ).with_inputs("trace")
        for r in records
    ]


def split(examples, train=0.2, dev=0.15, seed=0):
    """Split each MAS/benchmark/LLM group by the same fractions. The rest is test."""
    groups = defaultdict(list)
    for ex in examples:
        groups[ex.group].append(ex)
    rng = random.Random(seed)
    trainset, devset, testset = [], [], []
    for key in sorted(groups):
        items = groups[key]
        rng.shuffle(items)
        n_train, n_dev = round(len(items) * train), round(len(items) * dev)
        trainset += items[:n_train]
        devset += items[n_train : n_train + n_dev]
        testset += items[n_train + n_dev :]
    return trainset, devset, testset
