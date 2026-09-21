"""Five reporting rules, explicit coverage, and conditional assessment.

These tools do not establish experimental independence or confidence coverage.
The full receipt-based fixture workflow remains synthetic-only. The matrix API
accepts scored summaries under an explicit working model, without fitting models.
"""
from .api import publication_candidates, common_coverage, assessment_intervals as working_assessment_intervals
from .assessment_intervals import finite_panel_risk_bounds, paired_rule_difference_bounds
from .reporting_candidate import RULES

__all__ = ["RULES", "publication_candidates", "common_coverage", "working_assessment_intervals",
           "finite_panel_risk_bounds", "paired_rule_difference_bounds"]
