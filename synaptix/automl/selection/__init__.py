"""Model Selection Engine: registro de candidatos, evaluación y ranking."""

from .ranking import (
    CandidateResult,
    build_ranking,
    evaluate_candidate,
    evaluate_candidates,
    pick_best,
)
from .registry import (
    MODE_CANDIDATES,
    MODEL_REGISTRY,
    SPECS,
    Candidate,
    ModelSpec,
    get_spec,
    list_models,
    select_candidates,
)

__all__ = [
    "CandidateResult",
    "build_ranking",
    "evaluate_candidate",
    "evaluate_candidates",
    "pick_best",
    "MODE_CANDIDATES",
    "MODEL_REGISTRY",
    "SPECS",
    "Candidate",
    "ModelSpec",
    "get_spec",
    "list_models",
    "select_candidates",
]
