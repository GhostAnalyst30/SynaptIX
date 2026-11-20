"""Model Selection Engine: registro de candidatos, evaluación y ranking."""

from .feature_subset_search import FeatureSubsetSearch, count_combinations
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
    "FeatureSubsetSearch",
    "count_combinations",
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
