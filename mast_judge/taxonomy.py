"""The MAST failure taxonomy, written as Jev questions.

Each mode becomes one Noul. The `question` is the Noul's instruction and the
`true` and `false` entries are its criteria. Questions point into the request
state with backticked paths. DSPy places input fields under `inputs`.
"""

from dataclasses import dataclass

WINDOW = "`inputs.trace_window`"
TASK = "`inputs.task`"


@dataclass(frozen=True)
class Mode:
    code: str
    name: str
    category: str
    field: str
    question: str
    true: dict
    false: dict


CATEGORIES = {
    "1": "Specification issues",
    "2": "Inter-agent misalignment",
    "3": "Task verification",
}

MODES = [
    Mode(
        code="1.1",
        name="Disobey task specification",
        category="1",
        field="disobey_task_spec",
        question=f"In {WINDOW}, does an agent produce work that breaks a stated requirement or constraint of the task in {TASK}?",
        true={
            "what": "An output ignores or contradicts an explicit requirement: a required feature, input format, output format, language, library, or limit",
            "examples": [
                "The task asks for a GUI and the agents ship a command-line script",
                "The task says to answer in a boxed fraction and the agent gives a decimal",
                "The task asks to rate only liked songs and the agent rates every song",
            ],
        },
        false={
            "what": "Every output in the excerpt respects the requirements stated in the task, or the excerpt shows no output yet",
            "not_for": "Work that is correct in form but has a bug in its logic",
        },
    ),
    Mode(
        code="1.2",
        name="Disobey role specification",
        category="1",
        field="disobey_role_spec",
        question=f"In {WINDOW}, does an agent act outside the role it was assigned, doing another agent's job or refusing its own?",
        true={
            "what": "An agent takes over duties that belong to a different role, or declines the duty its role defines",
            "examples": [
                "A navigator asked to report code writes a new implementation instead",
                "A reviewer rewrites the whole program instead of reviewing it",
                "A CEO agent starts writing code",
            ],
        },
        false={"what": "Each agent stays within the responsibilities its role prompt describes"},
    ),
    Mode(
        code="1.3",
        name="Step repetition",
        category="1",
        field="step_repetition",
        question=f"In {WINDOW}, does an agent redo a step, message, or action that was already completed, without a new reason?",
        true={
            "what": "The same plan, thought, tool call, or phase appears again after it already finished, with no change in approach",
            "examples": [
                "The planner repeats exactly the same thought twice",
                "The agents run the same code review cycle again on unchanged code",
                "The same search query is issued again after it returned results",
            ],
        },
        false={
            "what": "Each step does new work",
            "not_for": "A retry that changes the approach after an error",
        },
    ),
    Mode(
        code="1.4",
        name="Loss of conversation history",
        category="1",
        field="loss_of_history",
        question=f"In {WINDOW}, does an agent lose track of earlier context and act as if recent messages or results never happened?",
        true={
            "what": "An agent reverts to an earlier state of the conversation, forgetting decisions, fixes, or facts established in recent turns",
            "examples": [
                "After a library was installed, the agent again says it is missing and swaps it out",
                "An agent asks for information that was given two turns earlier",
            ],
        },
        false={"what": "Agents build on the recent turns and keep established facts"},
    ),
    Mode(
        code="1.5",
        name="Unaware of termination conditions",
        category="1",
        field="unaware_of_termination",
        question=f"In {WINDOW}, do the agents keep going when they should stop, because they miss that the task is finished or cannot progress?",
        true={
            "what": "Agents continue past a stopping point: they keep prompting after the task is solved or after another agent says it cannot continue",
            "examples": [
                "An agent says the problem is unsolvable and the other replies 'Continue' again",
                "Agents keep polishing finished code until the turn limit ends the run",
                "The run stops only because it hits the maximum number of rounds",
            ],
        },
        false={"what": "Agents stop once the goal is met or once they know they cannot progress"},
    ),
    Mode(
        code="2.1",
        name="Conversation reset",
        category="2",
        field="conversation_reset",
        question=f"In {WINDOW}, does the conversation restart from the beginning without a reason, dropping the progress made so far?",
        true={
            "what": "The system reinitializes or begins the task again from scratch in the middle of a run",
            "examples": [
                "'Initialized HyperAgent instance' appears again partway through, followed by the same first thought",
                "The agents greet each other and restate the task as if starting fresh",
            ],
        },
        false={"what": "The conversation moves forward from where it was"},
    ),
    Mode(
        code="2.2",
        name="Fail to ask for clarification",
        category="2",
        field="fail_to_clarify",
        question=f"In {WINDOW}, does an agent proceed on unclear, missing, or contradictory information instead of asking for clarification?",
        true={
            "what": "An agent guesses or invents a missing detail when it could have asked another agent or the user",
            "examples": [
                "The agent lacks an access token and makes one up instead of asking for it",
                "The task is ambiguous about the input format and the agent picks one silently",
            ],
        },
        false={"what": "The information was clear, or the agent asked for what it lacked"},
    ),
    Mode(
        code="2.3",
        name="Task derailment",
        category="2",
        field="task_derailment",
        question=f"In {WINDOW}, does an agent drift away from the objective in {TASK} and work on something irrelevant or different?",
        true={
            "what": "An agent pursues a goal that does not serve the task",
            "examples": [
                "Asked to report an existing function, the agent invents a simplified version and modifies it",
                "An agent answers a different question than the one asked",
            ],
        },
        false={"what": "The work in the excerpt serves the objective of the task"},
    ),
    Mode(
        code="2.4",
        name="Information withholding",
        category="2",
        field="info_withholding",
        question=f"In {WINDOW}, does an agent hold important information that it fails to pass to the agent who needs it?",
        true={
            "what": "An agent knows a key fact, file, error, or result and does not share it, or reports it falsely",
            "examples": [
                "The bug localizer finds the faulty line but does not report it to the coder",
                "An agent reports an empty list of unimplemented files while files remain unimplemented",
            ],
        },
        false={"what": "Agents share the information others need"},
    ),
    Mode(
        code="2.5",
        name="Ignored other agent's input",
        category="2",
        field="ignored_input",
        question=f"In {WINDOW}, does an agent ignore a suggestion, correction, or warning from another agent?",
        true={
            "what": "One agent raises a point and the next agent continues without addressing it",
            "examples": [
                "An agent says the problem cannot be solved as stated and the other agent pushes ahead anyway",
                "A reviewer reports a bug and the programmer's next version still contains it unaddressed",
            ],
        },
        false={"what": "Agents respond to or act on the input they receive"},
    ),
    Mode(
        code="2.6",
        name="Action-reasoning mismatch",
        category="2",
        field="reasoning_action_mismatch",
        question=f"In {WINDOW}, does an agent's action or output contradict its own stated reasoning or conclusion?",
        true={
            "what": "What an agent does or says next does not follow from what it just reasoned or observed",
            "examples": [
                "The agent shows a method's code, then says the method is not shown",
                "The agent concludes it should run tests, then submits without running them",
            ],
        },
        false={"what": "Actions follow from the agent's stated reasoning"},
    ),
    Mode(
        code="3.1",
        name="Premature termination",
        category="3",
        field="premature_termination",
        question=f"In {WINDOW}, do the agents end the task before the objective is met or before needed information or checks are complete?",
        true={
            "what": "The run stops or declares completion while required work, data, or verification is still missing",
            "examples": [
                "The agent cannot get an access token and marks the task failed without asking the supervisor",
                "The coder writes code, the tester writes tests, and the run ends before the tests run",
            ],
        },
        false={
            "what": "The task is complete when the agents stop, or the excerpt does not show the agents stopping",
        },
    ),
    Mode(
        code="3.2",
        name="No or incomplete verification",
        category="3",
        field="no_or_incomplete_verification",
        question=f"In {WINDOW}, is a result accepted without a check, or does the checking stop before it covers the result?",
        true={
            "what": "No agent tests or reviews the result, or a planned check is skipped, never run, or left unfinished",
            "examples": [
                "The final answer is boxed without checking the arithmetic",
                "The tester writes tests and the run ends before anyone runs them",
                "The agent says the fix should work and submits without running the code",
            ],
        },
        false={"what": "Every result in the excerpt is checked, or the excerpt contains no result to check"},
    ),
    Mode(
        code="3.3",
        name="Incorrect verification",
        category="3",
        field="incorrect_verification",
        question=f"In {WINDOW}, does a check run but reach the wrong verdict or look only at the surface of the work?",
        true={
            "what": "A review or test exists, yet it approves work that is wrong, or it checks only syntax, format, or style and skips the logic and the task requirements",
            "examples": [
                "The reviewer approves the code although it does not compile",
                "The code reviewer comments on docstrings and misses that the Sudoku board starts empty",
                "The tester confirms the game runs but not that it announces the right winner",
            ],
        },
        false={
            "what": "The checks that run test the substance of the work and reach the right verdict",
            "not_for": "A result that is never checked at all",
        },
    ),
]

MODE_CODES = [m.code for m in MODES]
BY_FIELD = {m.field: m for m in MODES}
