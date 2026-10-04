"""Ladder questions: small literal Nouls that code combines into one MAST mode.

The ladders cover 1.2 (disobey role specification), 3.2 (no or incomplete
verification), and 3.3 (incorrect verification). Each rung asks one property of
a window. `combine` turns the rungs into a decision per mode, and `soft_scores`
turns them into one number per mode for threshold-free comparison.
"""

import dspy
from dspy.experimental import Noul

from mast_judge.signatures import TASK_DESC, WINDOW_INSTRUCTIONS
from mast_judge.taxonomy import TASK, WINDOW

# The modes where the ladder beat the single Noul in analysis/ladder_experiment.py.
LADDER_MODES = ("1.2", "3.2")

RUNGS = {
    # Verification.
    "check_present": (
        f"In {WINDOW}, does an agent review, test, or check a result that another agent or step produced?",
        {
            "what": "A review comment, a test run, a comparison against expected output, or an explicit check of an answer",
            "examples": ["The Code Reviewer comments on the code", "The tester runs pytest", "The agent verifies the sum by recomputing it"],
        },
        {"what": "No agent checks any result in the excerpt"},
    ),
    "check_executed": (
        f"In {WINDOW}, does an agent run code or tests and show the output of that run?",
        {
            "what": "The excerpt contains the printed output, test results, or error messages from an actual run",
            "examples": ["exitcode: 0 (execution succeeded) Code output: 42", "3 passed in 0.02s", "Traceback (most recent call last):"],
        },
        {"what": "Code or tests are only written or discussed, with no output from a run"},
    ),
    "check_requirements": (
        f"In {WINDOW}, does a review or test compare the result against the requirements stated in {TASK}?",
        {
            "what": "The check names a task requirement and tests whether the result meets it",
            "examples": ["The reviewer notes the Sudoku board must start with prefilled numbers", "The test checks that the winner announced is the player who got three in a row"],
        },
        {"what": "No check in the excerpt refers to what the task requires"},
    ),
    "check_surface": (
        f"In {WINDOW}, does a review or test look only at style, comments, formatting, imports, or syntax?",
        {
            "what": "The only points raised concern how the result looks, not whether it works or meets the task",
            "examples": ["The reviewer asks for more docstrings and approves", "The check confirms the file compiles and stops there"],
        },
        {"what": "The checks in the excerpt examine behavior or correctness, or there are no checks"},
    ),
    "wrong_pass": (
        f"In {WINDOW}, does an agent approve or accept a result even though the excerpt shows the result is wrong or broken?",
        {
            "what": "An error, failed test, or incorrect value is visible, and an agent still says the result is correct or done",
            "examples": ["The run prints a FileNotFoundError and the reviewer says the game works", "The computed answer contradicts the problem and the agent boxes it anyway"],
        },
        {"what": "Every result an agent accepts in the excerpt looks correct"},
    ),
    "unchecked_result": (
        f"In {WINDOW}, does an agent submit or announce a final result without running or reviewing it first?",
        {
            "what": "A final answer, program, or patch is delivered with no check of it in between",
            "examples": ["The agent writes the code and immediately says the task is complete", "The final answer is boxed right after the first calculation"],
        },
        {"what": "Each delivered result is checked first, or no final result appears in the excerpt"},
    ),
    "check_skipped": (
        f"In {WINDOW}, does an agent plan, write, or promise a test or review that is then not carried out?",
        {
            "what": "A check is set up or announced but never performed",
            "examples": ["The tester writes test cases and the run ends before they execute", "The agent says it will verify the answer and moves on without doing so"],
        },
        {"what": "Every check that is announced is carried out"},
    ),
    # Roles.
    "roles_defined": (
        f"Does {TASK} assign named roles or responsibilities to the agents?",
        {
            "what": "The opening names agents and says what each one is responsible for",
            "examples": ["You are a Code Reviewer...", "Alice(SimpleCoder), Bob(SimpleTester)", "The Navigator finds code and the Editor changes it"],
        },
        {"what": "The agents are not given distinct named roles"},
    ),
    "does_other_job": (
        f"In {WINDOW}, does an agent do work that the stated roles assign to a different agent?",
        {
            "what": "An agent takes over another role's duty",
            "examples": ["The Navigator, whose job is to find code, writes a new implementation", "The CEO agent writes the program", "The reviewer rewrites the whole file"],
        },
        {"what": "Each agent does the kind of work its own role describes"},
    ),
    "refuses_own_job": (
        f"In {WINDOW}, does an agent decline or fail to do the work its own role assigns to it?",
        {
            "what": "An agent says it cannot or will not do its duty, or hands back something other than what its role must produce",
            "examples": ["The Navigator says it can't provide the function's code and suggests installing the package instead"],
        },
        {"what": "Each agent attempts the duty its role assigns"},
    ),
    "speaks_as_other": (
        f"In {WINDOW}, does an agent write a message as if it were a different agent or the user?",
        {
            "what": "One agent produces turns that belong to another participant",
            "examples": ["The assistant writes both sides of the conversation", "The programmer replies on behalf of the reviewer"],
        },
        {"what": "Each agent speaks only for itself"},
    ),
}


def _signature():
    fields = {
        "task": (str, dspy.InputField(desc=TASK_DESC)),
        "window_position": (str, dspy.InputField(desc="Where the excerpt sits in the full log")),
        "trace_window": (str, dspy.InputField(desc="The excerpt of the log to judge")),
    }
    for name, (question, true, false) in RUNGS.items():
        fields[name] = (Noul[(True, true), (False, false)], dspy.OutputField(desc=question))
    return dspy.make_signature(fields, WINDOW_INSTRUCTIONS, signature_name="JudgeLadder")


JudgeLadder = _signature()


def combine(present: dict[str, bool]) -> dict[str, bool]:
    """Decide each mode from rungs that hold in at least one window."""
    return {
        "1.2": present["roles_defined"]
        and (present["does_other_job"] or present["refuses_own_job"] or present["speaks_as_other"]),
        "3.2": present["unchecked_result"] or present["check_skipped"] or not present["check_present"],
        "3.3": present["wrong_pass"] or (present["check_surface"] and not present["check_requirements"]),
    }


def soft_scores(p: dict[str, float]) -> dict[str, float]:
    """The same logic on probabilities: OR is max, AND is min, NOT is one minus."""
    return {
        "1.2": min(p["roles_defined"], max(p["does_other_job"], p["refuses_own_job"], p["speaks_as_other"])),
        "3.2": max(p["unchecked_result"], p["check_skipped"], 1 - p["check_present"]),
        "3.3": max(p["wrong_pass"], min(p["check_surface"], 1 - p["check_requirements"])),
    }
