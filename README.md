# MAST judge with DSPy and Jev

This project is a DSPy program that reads the log of a multi-agent system and marks which of the 14 [MAST](https://github.com/multi-agent-systems-failure-taxonomy/MAST) failure modes appear in it. The program asks its questions of Jev, the System One model from [TypeSafe](https://docs.typesafe.ai). Jev returns a probability for each answer instead of generated text. The program uses the experimental `TypeSafe` client, the `Noul`, `Choice`, and `Score` output types, and the `ReAnchor` optimizer from DSPy 3.4.0.

## Setup

Install the dependencies with uv.

```bash
uv sync
```

Put your TypeSafe key in a `.env` file at the project root.

```bash
TYPESAFE_API_KEY=...
```

## Usage

Run the optimizer. It downloads the traces, scores the judge before and after calibration, and saves the calibrated program to `runs/full/mast_judge.json`.

```bash
uv run python scripts/optimize.py --out runs/full
```

Judge one trace with the saved program.

```bash
uv run python scripts/judge.py path/to/trace.txt --program runs/full/mast_judge.json
```

Use the judge from Python.

```python
import dspy
from dspy.experimental import TypeSafe
from mast_judge.judge import MASTJudge

dspy.configure(lm=TypeSafe(model="jev-1.13.0"))
judge = MASTJudge(use_outcome=False)
judge.load("runs/full/mast_judge.json")

pred = judge(trace=open("trace.txt").read())
pred.failure_modes   # e.g. ["1.3", "1.5"]
pred.probabilities   # the highest probability for each mode across windows
```

Set `use_outcome` to match the saved program. `scripts/judge.py` reads it from the file.

## How the judge works

### Windows

A trace can be far longer than Jev accepts. Jev takes at most 32k tokens for the state plus the longest question. The traces in the dataset run from about 1k characters to 2M characters. The code splits each trace into windows of at most 40,000 characters, breaking only between lines. Some traces have about 2.6 characters per token, so a window stays under 16k tokens.

Each request to Jev holds three inputs.

- The window itself.
- The first 3,000 characters of the trace, where the task and the roles are usually stated.
- A sentence that says whether the window is the whole log, the opening, the middle, or the end.

The code also removes a ChatDev log line that repeats between most messages and carries no information.

### Questions

`JudgeWindow` in `mast_judge/signatures.py` asks one `Noul` per failure mode, all 14 in one request. A Noul is a yes or no question, and Jev returns the probability of yes. The questions and their criteria live in `mast_judge/taxonomy.py`. Each criterion has a short description and concrete examples drawn from the MAST definitions and examples.

A mode is present in a trace when any window passes that mode's threshold. This rule is the same as comparing the highest probability across windows to the threshold.

`JudgeOutcome` is an optional second request on the last window. It asks two questions.

- A `Choice` asks how the run ended: finished, stopped early, ran on past the point where it should stop, or unclear. "Stopped early" adds mode 3.1. "Ran on" adds mode 1.5.
- A `Score` asks how thoroughly the agents checked their final result, on four levels from no check to a test against the task requirements. The two lowest levels add mode 3.2. The third level adds mode 3.3.

### Calibration with ReAnchor

`ReAnchor` does not change the questions or add requests. It fits the numbers that turn Jev's probabilities into decisions.

- A threshold for each Noul.
- Cut points between the Score levels.
- A weight for each Choice option.

ReAnchor reruns the program for each candidate setting and keeps a setting only when it improves the metric across held-out folds of the training set. A calibration of the 14 Nouls takes about 600 passes over the training set.

`scripts/optimize.py` runs ReAnchor inside `replay()` from `mast_judge/replay.py`. The raw Jev answers do not change during a calibration. Only the settings that decode them change. Inside `replay()`, each predictor's raw answers stay in memory, keyed by its exact inputs, and DSPy decodes them again under the current settings. Without `replay()`, every call rebuilds the request, hashes it to find the cached answer, and copies it into the call history. `replay()` also swaps DSPy's check for direct `forward()` calls, which builds a full stack trace on every module call, for one that reads only the calling frame. The judge caches each trace's windows and skips its thread pool while replay is on.

With these changes, calibrating the Nouls takes 3 minutes instead of 22, and it fits the same thresholds. `analysis/reanchor_replay.py` checks this against an earlier run.

`scripts/optimize.py` calibrates two versions of the judge. One uses the window Nouls alone. The other adds the outcome Choice and Score. The script picks the version that scores higher on the dev split and reports both the uncalibrated and the calibrated judge on the test split.

## Data

The traces come from [mcemri/MAST-Data](https://huggingface.co/datasets/mcemri/MAST-Data). `MAD_full_dataset.json` has 1,642 traces from 7 multi-agent frameworks, each with a yes or no label for every mode. An LLM judge produced these labels, so the scores below measure agreement with that judge. They do not measure agreement with people.

The split keeps the same mix of framework, benchmark, and model in each part. It uses 20% of the traces for training, 15% for dev, and 65% for test.

The dataset also has 19 traces labeled by three people. Those labels use older versions of the taxonomy with 14 to 18 modes and different numbering, so this project does not use them.

Sources disagree on the names of modes 3.2 and 3.3. The dataset card and the judge prompt in the MAST repository treat 3.2 as a missing or incomplete check and 3.3 as a check that is wrong or only looks at the surface. `definitions.txt` in the MAST repository lists them in the opposite order. The questions here follow the dataset card.

## Results

These results come from one run of `scripts/optimize.py` with the default `trace_f1` metric. Calibration used 329 training traces. The dev split has 245 traces and the test split has 1,068.

On dev, the version with Nouls alone did slightly better than the version with the outcome questions, so the script chose it.

| Version on dev | Trace F1 | Macro F1 | Micro F1 |
|---|---|---|---|
| Nouls alone, uncalibrated | 0.456 | 0.525 | 0.568 |
| Nouls alone, calibrated | 0.543 | 0.434 | 0.618 |
| Nouls with Choice and Score, uncalibrated | 0.445 | 0.524 | 0.565 |
| Nouls with Choice and Score, calibrated | 0.536 | 0.421 | 0.608 |

On test, calibration raised trace F1 from 0.450 to 0.521.

| Mode | Name | Positive traces | F1 before | F1 after | Threshold |
|---|---|---|---|---|---|
| 1.1 | Disobey task specification | 386 | 0.68 | 0.68 | 0.50 |
| 1.2 | Disobey role specification | 65 | 0.25 | 0.00 | 0.90 |
| 1.3 | Step repetition | 421 | 0.85 | 0.84 | 0.32 |
| 1.4 | Loss of conversation history | 78 | 0.47 | 0.00 | 0.95 |
| 1.5 | Unaware of termination conditions | 277 | 0.81 | 0.81 | 0.47 |
| 2.1 | Conversation reset | 52 | 0.40 | 0.14 | 0.90 |
| 2.2 | Fail to ask for clarification | 170 | 0.41 | 0.42 | 0.53 |
| 2.3 | Task derailment | 234 | 0.63 | 0.63 | 0.16 |
| 2.4 | Information withholding | 15 | 0.10 | 0.00 | 0.96 |
| 2.5 | Ignored other agent's input | 240 | 0.58 | 0.58 | 0.50 |
| 2.6 | Action-reasoning mismatch | 393 | 0.59 | 0.66 | 0.27 |
| 3.1 | Premature termination | 183 | 0.43 | 0.30 | 0.91 |
| 3.2 | No or incomplete verification | 383 | 0.56 | 0.57 | 0.80 |
| 3.3 | Incorrect verification | 429 | 0.50 | 0.57 | 0.39 |
| | All modes, macro F1 | | 0.519 | 0.442 | |
| | All modes, micro F1 | | 0.561 | 0.614 | |

ReAnchor left the thresholds for 1.1 and 2.5 at 0.5 because no new value passed its fold check.

The judge agrees best with the labels on step repetition (1.3) and on runs that do not stop when they should (1.5). It agrees least on the rare modes. Under the `trace_f1` metric, ReAnchor raised the thresholds for 1.2, 1.4, 2.1, 2.4, and 3.1 to 0.9 or higher. The judge almost never predicts those modes after calibration.

## Ladder questions

`mast_judge/ladder.py` splits three modes into small literal Nouls, called rungs, and code combines the rungs into a decision. For example, 1.2 counts as present when the task assigns named roles and an agent does another role's work, refuses its own work, or speaks as another agent. `analysis/ladder_experiment.py` compares each ladder with the original question. It fits thresholds on train and dev to maximize each mode's F1 and scores the test split.

| Mode | Original F1 | Ladder F1 |
|---|---|---|
| 1.2 Disobey role specification | 0.32 | 0.42 |
| 3.2 No or incomplete verification | 0.58 | 0.63 |
| 3.3 Incorrect verification | 0.61 | 0.61 |

The ladder helped 1.2 and 3.2 and did nothing for 3.3. One rung did most of the work for 3.2. It asks whether an agent planned or promised a check that never happened. `MASTJudge(use_ladder=True)` uses the ladders for 1.2 and 3.2.

With `--metric balanced`, ReAnchor kept the rare modes on but flagged them far too often, e.g., 254 test traces for 2.4, which has 15 positives. Macro F1 on test stayed at 0.521 for both versions of the judge. Fitting each mode's threshold to that mode's own F1 gave a higher macro F1 of 0.543 in `analysis/per_mode_thresholds.py`.

## Metrics

ReAnchor needs a score for each trace.

- `trace_f1` is the default. It is the F1 between the true and predicted modes of one trace, and two empty sets score 1.
- `--metric balanced` weighs every mode equally. Its mean over the training set is the balanced accuracy averaged across the 14 modes.

The two metrics lead to different thresholds. `trace_f1` rewards the judge for never predicting a rare mode. `balanced` keeps rare modes in play at the cost of more false positives.
