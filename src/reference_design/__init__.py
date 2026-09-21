"""Reference operations, input evidence and fixed-scale scoring diagnostics."""

from .audit import plan_change, encode_prediction, decode_prediction, canonical_components
from .input_binding import verify_generation_record

from .core import (
    PATTERNS,
    ROLE_TUPLES,
    Allocation,
    Prediction,
    ScoreResult,
    allocate_controls,
    effect_to_state,
    pattern_for,
    score_references,
    state_to_effect,
)

__all__ = [
    "PATTERNS", "ROLE_TUPLES", "Allocation", "Prediction", "ScoreResult",
    "allocate_controls", "effect_to_state", "pattern_for", "score_references",
    "state_to_effect",
    "plan_change", "encode_prediction", "decode_prediction", "canonical_components",
    "verify_generation_record",
]
