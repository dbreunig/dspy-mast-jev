"""Reuse raw System One answers while ReAnchor calibrates a program.

ReAnchor scores each candidate setting by rerunning the program over the training set. Within
one calibration the raw answers never change. Only the thresholds, cuts, and weights that
decode them change. Inside `replay()`, each call's raw answers are kept in memory, keyed by the
calling predictor and its exact inputs, and decoded again under the current settings. The
program and the metric still run on every pass. A call with inputs not seen before goes through
the normal request path.

`replay()` also replaces DSPy's check for direct `forward()` calls, which builds a full stack
trace on every module call, with one that reads only the caller's frame.
"""

import contextlib
import logging
import sys
import threading

from dspy.adapters.decision import DecisionAdapter
from dspy.dsp.utils.settings import settings
from dspy.primitives.module import BaseModule, Module

logger = logging.getLogger("dspy.primitives.module")
_active = False


def active() -> bool:
    """Whether a `replay()` block is open. The patches apply to every thread."""
    return _active


def _getattribute(self, name):
    attr = BaseModule.__getattribute__(self, name)
    if name == "forward" and callable(attr) and sys._getframe(1).f_code.co_name != "__call__":
        logger.warning(
            f"Calling module.forward(...) on {self.__class__.__name__} directly is discouraged. "
            f"Please use module(...) instead."
        )
    return attr


@contextlib.contextmanager
def replay():
    """Serve repeated System One calls from raw answers held in memory."""
    global _active
    answers, lock = {}, threading.Lock()
    adapter_call, module_getattribute = DecisionAdapter.__call__, Module.__getattribute__

    def call(self, lm, lm_kwargs, signature, demos, inputs):
        if not self.system_one or demos or lm_kwargs:
            return adapter_call(self, lm, lm_kwargs, signature, demos, inputs)
        caller = (settings.caller_modules or [None])[-1]
        key = (
            id(caller),
            signature.instructions,
            tuple(signature.output_fields),
            tuple((k, inputs[k]) for k in signature.input_fields if k in inputs),
        )
        raw = answers.get(key)
        if raw is None:
            raw = lm(**self._prepare(signature, demos, inputs, lm_kwargs))
            with lock:
                answers[key] = raw
        return self._decode([raw])

    DecisionAdapter.__call__ = call
    Module.__getattribute__ = _getattribute
    _active = True
    try:
        yield
    finally:
        _active = False
        DecisionAdapter.__call__ = adapter_call
        Module.__getattribute__ = module_getattribute
