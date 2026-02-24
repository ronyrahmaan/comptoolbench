"""Evaluation module for CompToolBench.

Provides scoring, metrics, model adapters, and the evaluation runner.
"""

from comptoolbench.evaluation.matchers import match_arguments, match_value
from comptoolbench.evaluation.metrics import CompositionGapResult, compute_composition_gap
from comptoolbench.evaluation.model_adapter import (
    AVAILABLE_MODELS,
    ModelAdapter,
    ModelConfig,
    Provider,
    verify_all_providers,
    verify_model,
)
from comptoolbench.evaluation.runner import BenchmarkRunner, ModelRunResult, TaskResult
from comptoolbench.evaluation.scorers import CallScore, ModelCall, TaskScore, score_task

__all__ = [
    "AVAILABLE_MODELS",
    "BenchmarkRunner",
    "CallScore",
    "CompositionGapResult",
    "ModelAdapter",
    "ModelCall",
    "ModelConfig",
    "ModelRunResult",
    "Provider",
    "TaskResult",
    "TaskScore",
    "compute_composition_gap",
    "match_arguments",
    "match_value",
    "score_task",
    "verify_all_providers",
    "verify_model",
]
