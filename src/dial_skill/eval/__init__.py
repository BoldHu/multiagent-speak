"""Evaluation metrics, Blinded Dual-Judge, unified experiment runners, and paper report builders."""

from dial_skill.eval.judge import (
    BlindedDualJudge,
    compute_cohens_kappa,
    compute_spearman_rho,
    export_human_spotcheck_csv,
)
from dial_skill.eval.metrics import (
    bootstrap_ci,
    compute_ece,
    compute_pagemantle_metrics,
    compute_when2speak_metrics,
    evaluate_pilot_go_nogo,
    mcnemar_test,
)

__all__ = [
    "BlindedDualJudge",
    "bootstrap_ci",
    "compute_cohens_kappa",
    "compute_ece",
    "compute_pagemantle_metrics",
    "compute_spearman_rho",
    "compute_when2speak_metrics",
    "evaluate_pilot_go_nogo",
    "export_human_spotcheck_csv",
    "mcnemar_test",
]
