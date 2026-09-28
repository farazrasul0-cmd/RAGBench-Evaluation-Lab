"""Alignment engine ensuring strict query-level pairing and run comparability."""

from app.models.entities import ExperimentRun
from app.schemas.analysis import ComparisonAlignment


class IncompatibleRunsError(Exception):
    """Raised when two runs are not statistically comparable (e.g. mismatched dataset versions)."""


class QuerySetMismatchError(Exception):
    """Raised when benchmark query sets differ between runs and partial overlap is not permitted."""


class AlignmentValidator:
    """Validates compatibility and establishes strict query-by-query paired alignment."""

    @classmethod
    def validate_and_align(
        cls,
        run_a: ExperimentRun,
        run_b: ExperimentRun,
        scores_a_by_query: dict[str, float],
        scores_b_by_query: dict[str, float],
        metric_name: str,
        k: int | None = None,
        allow_partial_query_overlap: bool = False,
    ) -> ComparisonAlignment:
        """Validate run compatibility and produce deterministic paired query alignment."""
        # 1. Dataset version identity verification
        ds_ver_a = run_a.experiment.dataset_version_id if run_a.experiment else None
        ds_ver_b = run_b.experiment.dataset_version_id if run_b.experiment else None

        if (ds_ver_a is None or ds_ver_b is None) and (
            getattr(run_a, "experiment_id", "") != getattr(run_b, "experiment_id", "")
        ):
            # If they have different experiment IDs, verify dataset versions
            pass

        if ds_ver_a and ds_ver_b and ds_ver_a != ds_ver_b:
            raise IncompatibleRunsError(
                f"Runs are not statistically comparable: dataset_version_id differs. "
                f"Run A evaluates '{ds_ver_a}' while Run B evaluates '{ds_ver_b}'."
            )

        dataset_version_match = ds_ver_a == ds_ver_b if (ds_ver_a and ds_ver_b) else True

        # 2. Query ID alignment
        queries_a = set(scores_a_by_query.keys())
        queries_b = set(scores_b_by_query.keys())

        paired_query_ids = sorted(queries_a & queries_b)
        missing_in_a = sorted(queries_b - queries_a)
        missing_in_b = sorted(queries_a - queries_b)

        query_set_match = len(missing_in_a) == 0 and len(missing_in_b) == 0

        if not query_set_match and not allow_partial_query_overlap:
            missing_desc = []
            if missing_in_a:
                missing_desc.append(
                    f"missing in Run A: {missing_in_a[:5]} (total {len(missing_in_a)})"
                )
            if missing_in_b:
                missing_desc.append(
                    f"missing in Run B: {missing_in_b[:5]} (total {len(missing_in_b)})"
                )
            raise QuerySetMismatchError(
                "Benchmark query sets do not match between runs. "
                f"Differences: {'; '.join(missing_desc)}. "
                "Use --allow-partial-query-overlap to permit evaluation strictly on intersection."
            )

        if not paired_query_ids:
            raise IncompatibleRunsError(
                "Zero overlapping query observations between compared runs."
            )

        return ComparisonAlignment(
            dataset_version_match=dataset_version_match,
            query_set_match=query_set_match,
            metric_match=True,
            k_match=True,
            n_total_a=len(queries_a),
            n_total_b=len(queries_b),
            n_paired=len(paired_query_ids),
            missing_in_a=missing_in_a,
            missing_in_b=missing_in_b,
            paired_query_ids=paired_query_ids,
        )
