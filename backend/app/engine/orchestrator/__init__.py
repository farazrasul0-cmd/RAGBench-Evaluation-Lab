"""Orchestrator package for experiment matrix expansion, compatibility, and dry-run planning."""

from app.engine.orchestrator.cartesian import (
    CartesianExpander,
    ConfigurationExpansionError,
)
from app.engine.orchestrator.compatibility import (
    CompatibilityResult,
    RetrievalTopology,
    resolve_topology,
    validate_component_compatibility,
)
from app.engine.orchestrator.dry_run import (
    DryRunPlan,
    DryRunPlanner,
    PlannedPipelineRun,
)
from app.engine.orchestrator.identity import (
    CacheIdentity,
    compute_cache_identity,
)

__all__ = [
    "CacheIdentity",
    "CartesianExpander",
    "CompatibilityResult",
    "ConfigurationExpansionError",
    "DryRunPlan",
    "DryRunPlanner",
    "PlannedPipelineRun",
    "RetrievalTopology",
    "compute_cache_identity",
    "resolve_topology",
    "validate_component_compatibility",
]
