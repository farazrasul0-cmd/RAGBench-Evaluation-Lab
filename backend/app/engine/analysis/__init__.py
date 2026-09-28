"""Empirical research analysis, statistical hypothesis testing, and academic export package."""

from app.engine.analysis.alignment import (
    AlignmentValidator,
    IncompatibleRunsError,
    QuerySetMismatchError,
)
from app.engine.analysis.export import (
    AcademicLatexExporter,
    CSVExporter,
    ReplicationArchiveExporter,
    escape_latex,
)
from app.engine.analysis.statistics import (
    StatisticalComparator,
    benjamini_hochberg,
    holm_bonferroni,
    student_t_critical_value,
    student_t_two_tailed_p,
    wilcoxon_signed_rank_test,
)

__all__ = [
    "AcademicLatexExporter",
    "AlignmentValidator",
    "CSVExporter",
    "IncompatibleRunsError",
    "QuerySetMismatchError",
    "ReplicationArchiveExporter",
    "StatisticalComparator",
    "benjamini_hochberg",
    "escape_latex",
    "holm_bonferroni",
    "student_t_critical_value",
    "student_t_two_tailed_p",
    "wilcoxon_signed_rank_test",
]
