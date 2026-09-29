"""Alignment engine for query-level pairing, compatibility, and permutation invariance."""

import hashlib

from app.models.entities import ExperimentRun
from app.schemas.analysis import ComparisonAlignment


class IncompatibleRunsError(Exception):
    """Raised when two runs are not comparable (mismatched datasets, benchmarks, or K)."""


class QuerySetMismatchError(Exception):
    """Raised when benchmark query sets differ between runs and partial overlap is not permitted."""


class AlignmentValidator:
    """Validates compatibility and establishes deterministic paired query alignment."""

    @classmethod
    def validate_and_align(
        cls,
        run_a: ExperimentRun,
        run_b: ExperimentRun,
        scores_a_by_query: dict[str, float],
        scores_b_by_query: dict[str, float],
        metric_name: str,
        k: int | None = None,
        k_a: int | None = None,
        k_b: int | None = None,
        allow_partial_query_overlap: bool = False,
    ) -> ComparisonAlignment:
        """Validate compatibility, K parity, benchmark hash, and return alignment."""
        # 1. Dataset version identity verification
        ds_ver_a: str | None = None
        ds_hash_a: str | None = None
        bench_raw_a = getattr(run_a, "benchmark_hash", None)
        bench_hash_a: str | None = bench_raw_a if isinstance(bench_raw_a, str) else None
        try:
            exp_a = getattr(run_a, "experiment", None)
            if exp_a is not None:
                dva_id = getattr(exp_a, "dataset_version_id", None)
                if isinstance(dva_id, str):
                    ds_ver_a = dva_id
                if bench_hash_a is None:
                    bha = getattr(exp_a, "benchmark_hash", None)
                    if isinstance(bha, str):
                        bench_hash_a = bha
                dv_a = getattr(exp_a, "dataset_version", None)
                if dv_a is not None:
                    dha = getattr(dv_a, "content_hash", None)
                    if isinstance(dha, str):
                        ds_hash_a = dha
        except Exception:
            pass

        ds_ver_b: str | None = None
        ds_hash_b: str | None = None
        bench_raw_b = getattr(run_b, "benchmark_hash", None)
        bench_hash_b: str | None = bench_raw_b if isinstance(bench_raw_b, str) else None
        try:
            exp_b = getattr(run_b, "experiment", None)
            if exp_b is not None:
                dvb_id = getattr(exp_b, "dataset_version_id", None)
                if isinstance(dvb_id, str):
                    ds_ver_b = dvb_id
                if bench_hash_b is None:
                    bhb = getattr(exp_b, "benchmark_hash", None)
                    if isinstance(bhb, str):
                        bench_hash_b = bhb
                dv_b = getattr(exp_b, "dataset_version", None)
                if dv_b is not None:
                    dhb = getattr(dv_b, "content_hash", None)
                    if isinstance(dhb, str):
                        ds_hash_b = dhb
        except Exception:
            pass

        if ds_ver_a and ds_ver_b and ds_ver_a != ds_ver_b:
            raise IncompatibleRunsError(
                f"Runs are not statistically comparable: dataset_version_id differs. "
                f"Run A evaluates '{ds_ver_a}' while Run B evaluates '{ds_ver_b}'."
            )

        if ds_hash_a and ds_hash_b and ds_hash_a != ds_hash_b:
            raise IncompatibleRunsError(
                f"Runs are not statistically comparable: dataset content_hash differs. "
                f"Run A evaluates '{ds_hash_a}' while Run B evaluates '{ds_hash_b}'."
            )

        # 2. Benchmark specification hash verification
        if bench_hash_a and bench_hash_b and bench_hash_a != bench_hash_b:
            raise IncompatibleRunsError(
                f"Runs are not statistically comparable: benchmark specification hash differs. "
                f"Run A evaluates '{bench_hash_a}' while Run B evaluates '{bench_hash_b}'."
            )

        # 2b. Evaluation protocol & metric definition version compatibility
        raw_eval_a = getattr(run_a, "evaluation_protocol_version", None)
        if not isinstance(raw_eval_a, str) and hasattr(run_a, "experiment") and run_a.experiment:
            raw_eval_a = getattr(run_a.experiment, "evaluation_protocol_version", None)
        proto_eval_a = raw_eval_a if isinstance(raw_eval_a, str) else None

        raw_eval_b = getattr(run_b, "evaluation_protocol_version", None)
        if not isinstance(raw_eval_b, str) and hasattr(run_b, "experiment") and run_b.experiment:
            raw_eval_b = getattr(run_b.experiment, "evaluation_protocol_version", None)
        proto_eval_b = raw_eval_b if isinstance(raw_eval_b, str) else None

        raw_metric_a = getattr(run_a, "metric_definition_version", None)
        if not isinstance(raw_metric_a, str) and hasattr(run_a, "experiment") and run_a.experiment:
            raw_metric_a = getattr(run_a.experiment, "metric_definition_version", None)
        proto_metric_a = raw_metric_a if isinstance(raw_metric_a, str) else None

        raw_metric_b = getattr(run_b, "metric_definition_version", None)
        if not isinstance(raw_metric_b, str) and hasattr(run_b, "experiment") and run_b.experiment:
            raw_metric_b = getattr(run_b.experiment, "metric_definition_version", None)
        proto_metric_b = raw_metric_b if isinstance(raw_metric_b, str) else None

        if proto_eval_a and proto_eval_b and proto_eval_a != proto_eval_b:
            raise IncompatibleRunsError(
                "Runs are not statistically comparable: evaluation_protocol_version differs. "
                f"Run A evaluates with '{proto_eval_a}' while Run B evaluates with "
                f"'{proto_eval_b}'."
            )

        if proto_metric_a and proto_metric_b and proto_metric_a != proto_metric_b:
            raise IncompatibleRunsError(
                "Runs are not statistically comparable: metric_definition_version differs. "
                f"Run A evaluates with '{proto_metric_a}' while Run B evaluates with "
                f"'{proto_metric_b}'."
            )

        # 3. Metric parameter K compatibility
        eff_k_a = k_a if k_a is not None else k
        eff_k_b = k_b if k_b is not None else k
        if eff_k_a is not None and eff_k_b is not None and eff_k_a != eff_k_b:
            raise IncompatibleRunsError(
                f"Runs are not statistically comparable: metric K differs for '{metric_name}'. "
                f"Run A evaluated at K={eff_k_a} while Run B evaluated at K={eff_k_b}."
            )
        # 4. Query set alignment and permutation-invariant pairing
        queries_a = set(scores_a_by_query.keys())
        queries_b = set(scores_b_by_query.keys())

        # Check for duplicate query keys in input observations
        if len(queries_a) != len(scores_a_by_query):
            raise ValueError("Duplicate query IDs detected in Run A observations.")
        if len(queries_b) != len(scores_b_by_query):
            raise ValueError("Duplicate query IDs detected in Run B observations.")

        query_hash_a = hashlib.sha256(",".join(sorted(queries_a)).encode("utf-8")).hexdigest()
        query_hash_b = hashlib.sha256(",".join(sorted(queries_b)).encode("utf-8")).hexdigest()

        # Deterministic lexicographical sort ensures 100% permutation invariance
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
            raise ValueError(
                f"Zero overlapping queries between Run A and Run B for metric '{metric_name}'."
            )

        clean_ds_hash_a = ds_hash_a if isinstance(ds_hash_a, str) else None
        clean_ds_hash_b = ds_hash_b if isinstance(ds_hash_b, str) else None
        clean_bench_hash_a = bench_hash_a if isinstance(bench_hash_a, str) else None
        clean_bench_hash_b = bench_hash_b if isinstance(bench_hash_b, str) else None
        clean_ds_ver_a = ds_ver_a if isinstance(ds_ver_a, str) else ""
        clean_ds_ver_b = ds_ver_b if isinstance(ds_ver_b, str) else ""

        return ComparisonAlignment(
            dataset_version_match=clean_ds_ver_a == clean_ds_ver_b
            if (clean_ds_ver_a and clean_ds_ver_b)
            else True,
            dataset_version_id_a=clean_ds_ver_a,
            dataset_version_id_b=clean_ds_ver_b,
            dataset_version_hash_a=clean_ds_hash_a,
            dataset_version_hash_b=clean_ds_hash_b,
            benchmark_match=clean_bench_hash_a == clean_bench_hash_b
            if (clean_bench_hash_a and clean_bench_hash_b)
            else True,
            benchmark_hash_a=clean_bench_hash_a,
            benchmark_hash_b=clean_bench_hash_b,
            query_set_match=query_set_match,
            query_set_hash_a=query_hash_a,
            query_set_hash_b=query_hash_b,
            metric_name=metric_name,
            k_a=eff_k_a,
            k_b=eff_k_b,
            metric_match=True,
            k_match=eff_k_a == eff_k_b if (eff_k_a is not None and eff_k_b is not None) else True,
            n_total_a=len(queries_a),
            n_total_b=len(queries_b),
            n_paired=len(paired_query_ids),
            missing_in_a=missing_in_a,
            missing_in_b=missing_in_b,
            paired_query_ids=paired_query_ids,
            protocol_match=(
                (proto_eval_a == proto_eval_b if (proto_eval_a and proto_eval_b) else True)
                and (
                    proto_metric_a == proto_metric_b
                    if (proto_metric_a and proto_metric_b)
                    else True
                )
            ),
            evaluation_protocol_version_a=str(proto_eval_a or ""),
            evaluation_protocol_version_b=str(proto_eval_b or ""),
            metric_definition_version_a=str(proto_metric_a or ""),
            metric_definition_version_b=str(proto_metric_b or ""),
        )
