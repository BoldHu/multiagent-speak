"""Publication-grade metrics: When2Speak (Macro F1, FIR, MIR, AUROC, AUPRC, ECE, McNemar, Bootstrap CI) & PageMantle."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from dial_skill.schemas import (
    TIER_TO_INT,
    EpisodeResult,
    InterventionTier,
    JudgeRubricScore,
    PageMantleTask,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def fit_platt_parameters(
    y_true: Sequence[int | bool],
    scores: Sequence[float],
) -> tuple[float, float]:
    """Fit 1D Platt scaling `P(y=1|s) = sigmoid(a * s + b)` strictly on validation data."""
    y = np.asarray([int(bool(v)) for v in y_true], dtype=int)
    s = np.asarray(scores, dtype=float).reshape(-1, 1)
    if len(y) == 0:
        return 1.0, 0.0
    pos_rate = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    if len(set(y.tolist())) < 2 or float(np.std(s)) < 1e-8:
        return 0.0, float(math.log(pos_rate / (1.0 - pos_rate)))
    clf = LogisticRegression(C=50.0, solver="lbfgs", max_iter=500)
    clf.fit(s, y)
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def apply_platt_scaling(
    scores: Sequence[float],
    slope_a: float,
    intercept_b: float,
) -> np.ndarray:
    """Apply validation-fitted Platt parameters `(a, b)` to map raw margins to calibrated probabilities."""
    s = np.asarray(scores, dtype=float)
    return _sigmoid(slope_a * s + intercept_b)


def compute_ece(
    y_true: Sequence[int | bool],
    probs: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) over probability predictions in [0, 1]."""
    y = np.asarray([int(bool(v)) for v in y_true], dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc = float(np.mean(y[mask]))
            conf = float(np.mean(p[mask]))
            ece += (cnt / n) * abs(acc - conf)
    return round(float(ece), 6)


def mcnemar_test(
    y_true: Sequence[int | bool],
    pred_a: Sequence[int | bool],
    pred_b: Sequence[int | bool],
) -> dict[str, float | int]:
    """Exact two-sided McNemar's test comparing paired binary classifiers A vs. B."""
    yt = [bool(x) for x in y_true]
    pa = [bool(x) for x in pred_a]
    pb = [bool(x) for x in pred_b]
    if not (len(yt) == len(pa) == len(pb)):
        raise ValueError("mcnemar_test requires equal-length sequences.")

    n_a_only = sum(1 for y, a, b in zip(yt, pa, pb) if (a == y) and (b != y))
    n_b_only = sum(1 for y, a, b in zip(yt, pa, pb) if (a != y) and (b == y))
    n_disc = n_a_only + n_b_only
    if n_disc == 0:
        return {
            "n_a_correct_b_wrong": 0,
            "n_b_correct_a_wrong": 0,
            "chi2_stat": 0.0,
            "p_value": 1.0,
        }
    chi2 = ((abs(n_a_only - n_b_only) - 1.0) ** 2) / n_disc if abs(n_a_only - n_b_only) >= 1 else 0.0
    pval = float(binomtest(min(n_a_only, n_b_only), n_disc, 0.5, alternative="two-sided").pvalue)
    return {
        "n_a_correct_b_wrong": n_a_only,
        "n_b_correct_a_wrong": n_b_only,
        "chi2_stat": round(float(chi2), 6),
        "p_value": round(pval, 8),
    }


def paired_bootstrap_test(
    y_true: Sequence[int | bool],
    pred_a: Sequence[int | bool],
    pred_b: Sequence[int | bool],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap hypothesis test & 95% CI directly on Delta Macro-F1 = F1(A) - F1(B)."""
    yt = np.asarray([int(bool(x)) for x in y_true], dtype=int)
    pa = np.asarray([int(bool(x)) for x in pred_a], dtype=int)
    pb = np.asarray([int(bool(x)) for x in pred_b], dtype=int)
    if not (len(yt) == len(pa) == len(pb)):
        raise ValueError("paired_bootstrap_test requires equal-length sequences.")
    n = len(yt)
    if n == 0:
        return {"delta_macro_f1": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value": 1.0}

    obs_a = float(f1_score(yt, pa, average="macro", zero_division=0))
    obs_b = float(f1_score(yt, pb, average="macro", zero_division=0))
    obs_delta = obs_a - obs_b

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        fa = float(f1_score(yt[idx], pa[idx], average="macro", zero_division=0))
        fb = float(f1_score(yt[idx], pb[idx], average="macro", zero_division=0))
        deltas[i] = fa - fb

    ci_lo = float(np.quantile(deltas, 0.025))
    ci_hi = float(np.quantile(deltas, 0.975))
    # Two-sided paired bootstrap p-value via centered shift / tail mass
    p_le = float(np.mean(deltas <= 0.0))
    p_ge = float(np.mean(deltas >= 0.0))
    pval = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "macro_f1_a": round(obs_a, 6),
        "macro_f1_b": round(obs_b, 6),
        "delta_macro_f1": round(obs_delta, 6),
        "ci95_low": round(ci_lo, 6),
        "ci95_high": round(ci_hi, 6),
        "p_value": round(pval, 8),
    }


def paired_nonparametric_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired sign-flip permutation test + Wilcoxon signed-rank test + 95% CI on task-level scores."""
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    if len(sa) != len(sb):
        raise ValueError("paired_nonparametric_test requires equal-length score arrays.")
    n = len(sa)
    if n == 0:
        return {
            "mean_diff": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "permutation_p_value": 1.0,
            "wilcoxon_p_value": 1.0,
        }
    diffs = sa - sb
    obs_diff = float(np.mean(diffs))
    ci = bootstrap_ci(diffs, n_bootstrap=2000, seed=seed)

    # Exact sign-flip enumeration when n <= 16, else Monte Carlo sign-flip permutation
    if n <= 16:
        signs = np.array(
            [[1.0 if ((mask >> j) & 1) else -1.0 for j in range(n)] for mask in range(1 << n)],
            dtype=float,
        )
        perm_means = (signs @ diffs) / float(n)
        perm_p = float(np.mean(np.abs(perm_means) >= abs(obs_diff) - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
        perm_means = (signs @ diffs) / float(n)
        perm_p = float((np.sum(np.abs(perm_means) >= abs(obs_diff) - 1e-12) + 1) / (n_permutations + 1))

    non_zero = diffs[np.abs(diffs) > 1e-12]
    if len(non_zero) < 2:
        wilcox_p = 1.0 if len(non_zero) == 0 else 0.5
    else:
        try:
            wilcox_p = float(wilcoxon(non_zero, alternative="two-sided").pvalue)
        except Exception:
            wilcox_p = perm_p

    return {
        "mean_a": round(float(np.mean(sa)), 4),
        "mean_b": round(float(np.mean(sb)), 4),
        "mean_diff": round(obs_diff, 4),
        "ci95_low": round(ci["ci_low"], 4),
        "ci95_high": round(ci["ci_high"], 4),
        "permutation_p_value": round(min(1.0, perm_p), 8),
        "wilcoxon_p_value": round(min(1.0, wilcox_p), 8),
    }


def holm_bonferroni_correction(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    """Step-down Holm-Bonferroni family-wise error rate correction over named hypothesis tests."""
    m = len(p_values)
    if m == 0:
        return {}
    ordered = sorted(p_values.items(), key=lambda kv: float(kv[1]))
    adjusted_map: dict[str, dict[str, float | bool]] = {}
    running_max = 0.0
    still_rejecting = True
    for rank_idx, (name, raw_p) in enumerate(ordered):
        mult = m - rank_idx
        adj_p = min(1.0, max(running_max, float(raw_p) * mult))
        running_max = adj_p
        if adj_p > alpha:
            still_rejecting = False
        adjusted_map[name] = {
            "raw_p_value": round(float(raw_p), 8),
            "holm_adjusted_p_value": round(adj_p, 8),
            "significant_at_alpha": bool(still_rejecting),
        }
    return {k: adjusted_map[k] for k in p_values}


def bootstrap_ci(
    values_a: Sequence[Any],
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    values_b: Sequence[Any] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Deterministic paired or univariate bootstrap confidence interval."""
    arr_a = np.asarray(values_a)
    n = len(arr_a)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)

    if values_b is not None and metric_fn is not None:
        arr_b = np.asarray(values_b)
        point = float(metric_fn(arr_a, arr_b))
        samples = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            samples[i] = float(metric_fn(arr_a[idx], arr_b[idx]))
    else:
        arr_f = arr_a.astype(float)
        point = float(np.mean(arr_f))
        samples = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            samples[i] = float(np.mean(arr_f[idx]))

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return {
        "point": round(point, 6),
        "ci_low": round(lo, 6),
        "ci_high": round(hi, 6),
    }


def compute_when2speak_metrics(
    y_true: Sequence[bool],
    episodes: Sequence[EpisodeResult],
    n_bootstrap: int = 1000,
    seed: int = 42,
    calibrated_probs: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute full When2Speak metric suite (Macro F1, SPEAK F1/P/R, FIR, MIR, AUROC, AUPRC, ECE, Token Reduction)."""
    if len(y_true) != len(episodes):
        raise ValueError("y_true and episodes must have the same length.")
    if not episodes:
        return {"n_samples": 0}

    yt = np.asarray([int(bool(y)) for y in y_true], dtype=int)
    yp = np.asarray([int(bool(ep.gate.binary_speak)) for ep in episodes], dtype=int)
    margins = np.asarray([float(ep.gate.margin_delta) for ep in episodes], dtype=float)
    probs = (
        np.clip(np.asarray(calibrated_probs, dtype=float), 0.0, 1.0)
        if calibrated_probs is not None
        else _sigmoid(margins * 3.0)
    )

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    fir = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    mir = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    speak_f1 = float(f1_score(yt, yp, pos_label=1, zero_division=0))
    speak_p = float(precision_score(yt, yp, pos_label=1, zero_division=0))
    speak_r = float(recall_score(yt, yp, pos_label=1, zero_division=0))
    silent_f1 = float(f1_score(yt, yp, pos_label=0, zero_division=0))

    has_both_classes = len(set(yt.tolist())) == 2
    auroc = float(roc_auc_score(yt, margins)) if has_both_classes else 0.5
    auprc = float(average_precision_score(yt, margins)) if has_both_classes else 0.0
    ece = compute_ece(yt, probs)

    pos_margins = margins[yt == 1]
    neg_margins = margins[yt == 0]
    margin_gap = (
        float(np.mean(pos_margins) - np.mean(neg_margins))
        if (len(pos_margins) > 0 and len(neg_margins) > 0)
        else 0.0
    )

    raw_tok = sum(ep.prompt_tokens_raw for ep in episodes)
    slate_tok = sum(ep.prompt_tokens_slate for ep in episodes)
    token_reduction = (1.0 - (slate_tok / raw_tok)) if raw_tok > 0 else 0.0
    short_circuit_rate = sum(1 for ep in episodes if ep.gate.short_circuited) / len(episodes)
    billed_cost = sum(ep.total_cost_usd for ep in episodes)
    total_cost = sum(
        (ep.uncached_cost_usd if getattr(ep, "uncached_cost_usd", 0.0) > 0.0 else ep.total_cost_usd)
        for ep in episodes
    )

    macro_ci = bootstrap_ci(
        yt,
        metric_fn=lambda a, b: float(f1_score(a, b, average="macro", zero_division=0)),
        values_b=yp,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return {
        "n_samples": len(episodes),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "macro_f1": round(macro_f1, 6),
        "macro_f1_ci95": [macro_ci["ci_low"], macro_ci["ci_high"]],
        "speak_f1": round(speak_f1, 6),
        "speak_precision": round(speak_p, 6),
        "speak_recall": round(speak_r, 6),
        "silent_f1": round(silent_f1, 6),
        "fir": round(fir, 6),
        "mir": round(mir, 6),
        "auroc": round(auroc, 6),
        "auprc": round(auprc, 6),
        "ece": round(ece, 6),
        "ece_is_platt_calibrated": calibrated_probs is not None,
        "margin_gap": round(margin_gap, 6),
        "token_reduction_pct": round(token_reduction * 100.0, 2),
        "short_circuit_rate": round(short_circuit_rate, 4),
        "total_cost_usd": round(total_cost, 6),
        "billed_cost_usd": round(billed_cost, 6),
        "cost_per_1k_turns_usd": round(total_cost * 1000.0 / len(episodes), 4),
    }


def _deescalation_utility(pred_tier: InterventionTier, ideal_tier: InterventionTier) -> float:
    """Asymmetric channel utility in [0, 1]: penalizes public overreach harder than gentle de-escalation."""
    p = TIER_TO_INT[pred_tier]
    g = TIER_TO_INT[ideal_tier]
    if p == g:
        return 1.0
    if g == 0:
        # Ground truth SILENT: DM interruption (0.25) hurts far less than public channel blast (0.0)
        return {1: 0.25, 2: 0.10, 3: 0.0}[p]
    if p == 0:
        return 0.0  # Missed intervention
    diff = p - g
    if diff < 0:
        # Polite de-escalation (e.g., DM instead of Thread) preserves 0.75 utility
        return max(0.4, 1.0 - 0.25 * abs(diff))
    # Scope overreach (e.g., Channel blast instead of DM) penalized sharply
    return max(0.0, 1.0 - 0.40 * diff)


def compute_pagemantle_metrics(
    tasks: Sequence[PageMantleTask],
    episodes: Sequence[EpisodeResult],
    judge_scores: Sequence[JudgeRubricScore] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute PageMantle 4-dimension rubric aggregation, Silence Acc, Action Recall, Channel Match & Cost."""
    task_map = {t.task_id: t for t in tasks}
    judge_map = {j.sample_id: j for j in (judge_scores or [])}

    action_total = 0
    action_hits = 0
    trap_total = 0
    trap_hits = 0
    channel_exact = 0
    deesc_utils: list[float] = []

    rubric_axes: dict[str, list[float]] = {
        "intervention_or_silence": [],
        "content_and_evidence": [],
        "clarity_and_tone": [],
        "audience_and_location": [],
        "overall_score": [],
    }
    by_category: dict[str, list[float]] = {}

    for ep in episodes:
        task = task_map.get(ep.sample_id)
        if task is None:
            continue
        pred_speak = ep.gate.binary_speak
        if task.expected_action == "SILENT":
            trap_total += 1
            if not pred_speak:
                trap_hits += 1
        else:
            action_total += 1
            if pred_speak:
                action_hits += 1

        if ep.gate.final_tier == task.ideal_tier:
            channel_exact += 1
        deesc_utils.append(_deescalation_utility(ep.gate.final_tier, task.ideal_tier))

        j = judge_map.get(ep.sample_id)
        if j is not None:
            rubric_axes["intervention_or_silence"].append(j.intervention_or_silence)
            rubric_axes["content_and_evidence"].append(j.content_and_evidence)
            rubric_axes["clarity_and_tone"].append(j.clarity_and_tone)
            rubric_axes["audience_and_location"].append(j.audience_and_location)
            rubric_axes["overall_score"].append(j.overall_score)
            by_category.setdefault(task.category, []).append(j.overall_score)

    n = len(episodes)
    raw_tok = sum(ep.prompt_tokens_raw for ep in episodes)
    slate_tok = sum(ep.prompt_tokens_slate for ep in episodes)
    token_red = (1.0 - (slate_tok / raw_tok)) if raw_tok > 0 else 0.0
    timeout_rate = sum(1 for ep in episodes if ep.timed_out) / max(1, n)
    avg_tools = sum(ep.tool_calls_used for ep in episodes) / max(1, n)
    billed_cost = sum(ep.total_cost_usd for ep in episodes)
    total_cost = sum(
        (ep.uncached_cost_usd if getattr(ep, "uncached_cost_usd", 0.0) > 0.0 else ep.total_cost_usd)
        for ep in episodes
    )

    overall_scores = rubric_axes["overall_score"]
    overall_ci = (
        bootstrap_ci(overall_scores, n_bootstrap=n_bootstrap, seed=seed)
        if overall_scores
        else {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    )

    return {
        "n_tasks": n,
        "overall_score": round(float(np.mean(overall_scores)), 4) if overall_scores else 0.0,
        "overall_score_ci95": [round(overall_ci["ci_low"], 4), round(overall_ci["ci_high"], 4)],
        "intervention_or_silence": round(float(np.mean(rubric_axes["intervention_or_silence"])), 4)
        if rubric_axes["intervention_or_silence"]
        else 0.0,
        "content_and_evidence": round(float(np.mean(rubric_axes["content_and_evidence"])), 4)
        if rubric_axes["content_and_evidence"]
        else 0.0,
        "clarity_and_tone": round(float(np.mean(rubric_axes["clarity_and_tone"])), 4)
        if rubric_axes["clarity_and_tone"]
        else 0.0,
        "audience_and_location": round(float(np.mean(rubric_axes["audience_and_location"])), 4)
        if rubric_axes["audience_and_location"]
        else 0.0,
        "silence_accuracy": round(trap_hits / trap_total, 4) if trap_total > 0 else 1.0,
        "action_recall": round(action_hits / action_total, 4) if action_total > 0 else 1.0,
        "channel_match_rate": round(channel_exact / max(1, n), 4),
        "deescalation_utility": round(float(np.mean(deesc_utils)), 4) if deesc_utils else 0.0,
        "timeout_rate": round(timeout_rate, 4),
        "avg_tool_calls": round(avg_tools, 2),
        "token_reduction_pct": round(token_red * 100.0, 2),
        "total_cost_usd": round(total_cost, 6),
        "billed_cost_usd": round(billed_cost, 6),
        "cost_per_task_usd": round(total_cost / max(1, n), 6),
        "per_category_overall": {
            cat: round(float(np.mean(vals)), 4) for cat, vals in sorted(by_category.items())
        },
    }


def evaluate_pilot_go_nogo(
    *,
    w2s_dial: dict[str, Any],
    w2s_direct: dict[str, Any],
    pm_dial: dict[str, Any],
    pm_direct: dict[str, Any],
    triage_long_chat_token_reduction: float,
    triage_decisive_recall: float,
    dbcse_anchor_regressions: int,
    dbcse_heldout_delta: float,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Automated Phase-1 Pilot GO / NO-GO Scorecard matching docs/TECHNICAL_DESIGN.md Section 6."""
    th = {
        "w2s_min_macro_f1": 0.62,
        "w2s_min_delta_vs_direct": 0.08,
        "w2s_max_fir": 0.12,
        "w2s_max_mir": 0.45,
        "pm_min_delta_vs_direct": 0.60,
        "pm_min_silence_acc": 0.75,
        "pm_max_timeout_rate": 0.125,
        "triage_min_token_reduction": 0.40,
        "triage_min_decisive_recall": 0.90,
        "margin_min_gap": 0.20,
        "margin_min_auroc": 0.78,
        "dbcse_min_heldout_delta": -0.05,
        **(thresholds or {}),
    }

    c1 = (
        w2s_dial.get("macro_f1", 0.0) >= w2s_direct.get("macro_f1", 0.0) + th["w2s_min_delta_vs_direct"]
        and w2s_dial.get("macro_f1", 0.0) >= th["w2s_min_macro_f1"]
        and w2s_dial.get("fir", 1.0) <= th["w2s_max_fir"]
        and w2s_dial.get("mir", 1.0) <= th["w2s_max_mir"]
    )
    c2 = (
        pm_dial.get("overall_score", 0.0) >= pm_direct.get("overall_score", 0.0) + th["pm_min_delta_vs_direct"]
        and pm_dial.get("silence_accuracy", 0.0) >= th["pm_min_silence_acc"]
        and pm_dial.get("timeout_rate", 1.0) <= th["pm_max_timeout_rate"]
    )
    c3 = (
        triage_long_chat_token_reduction >= th["triage_min_token_reduction"]
        and triage_decisive_recall >= th["triage_min_decisive_recall"]
    )
    c4 = (
        w2s_dial.get("margin_gap", 0.0) >= th["margin_min_gap"]
        and w2s_dial.get("auroc", 0.0) >= th["margin_min_auroc"]
    )
    c5 = (
        dbcse_anchor_regressions == 0
        and dbcse_heldout_delta >= th["dbcse_min_heldout_delta"]
    )

    checks = {
        "1_when2speak_pilot_timing": c1,
        "2_pagemantle_end_to_end": c2,
        "3_jev_triage_compression": c3,
        "4_margin_discriminability": c4,
        "5_dbcse_sanity": c5,
    }
    passed_count = sum(1 for v in checks.values() if v)
    verdict = "GO" if (c1 and c2 and passed_count >= 4) else "NO_GO"

    diagnostics: list[str] = []
    if w2s_dial.get("fir", 0.0) > th["w2s_max_fir"]:
        diagnostics.append("FIR too high: raise lambda (1.0 -> 1.25) or tau_dm (0.05 -> 0.12).")
    if w2s_dial.get("mir", 0.0) > th["w2s_max_mir"]:
        diagnostics.append("MIR too high: lower tau_dm (0.05 -> -0.05), theta_skip (0.15 -> 0.08), or theta_floor.")
    if pm_dial.get("timeout_rate", 0.0) > th["pm_max_timeout_rate"]:
        diagnostics.append("PageMantle timeout rate high: tighten max_tool_calls to 5 and B_recon to 2.")
    if not math.isfinite(w2s_dial.get("auroc", 0.0)):
        diagnostics.append("Check margin distribution for degenerate scores.")

    return {
        "verdict": verdict,
        "passed_checks": passed_count,
        "total_checks": 5,
        "checks": checks,
        "diagnostics": diagnostics,
    }
