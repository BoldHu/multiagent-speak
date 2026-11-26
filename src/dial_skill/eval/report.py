"""Publication-ready LaTeX table builders, matplotlib vector PDF figures, and case study extractor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dial_skill.schemas import EpisodeResult, InterventionTier, PageMantleTask

# Clean publication-ready styling
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )


def _fmt(val: Any, digits: int = 3) -> str:
    if val is None or val == "":
        return "--"
    if isinstance(val, (int, float, np.floating, np.integer)):
        return f"{float(val):.{digits}f}"
    return _tex_escape(str(val))


# ---------------------------------------------------------------------------
# 1. LaTeX Table Generators (4 Publication Tables)
# ---------------------------------------------------------------------------


def generate_tab_when2speak(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate tab_when2speak.tex (Exp 1: When2Speak-Main timing table)."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main timing results on \\textbf{When2Speak-Main} (2,000 turns). DIAL-Skill achieves strong Macro F1 with zero weight updates and continuous FIR--MIR control.}",
        "\\label{tab:when2speak}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Model} & \\textbf{Macro F1} $\\uparrow$ & \\textbf{SPEAK F1} $\\uparrow$ & \\textbf{FIR} $\\downarrow$ & \\textbf{MIR} $\\downarrow$ & \\textbf{AUROC} $\\uparrow$ & \\textbf{Tok Red. (\\%)} $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r.get('method', ''))} & {_tex_escape(r.get('model', ''))} & "
            f"{_fmt(r.get('macro_f1'), 3)} & {_fmt(r.get('speak_f1'), 3)} & "
            f"{_fmt(r.get('fir'), 3)} & {_fmt(r.get('mir'), 3)} & "
            f"{_fmt(r.get('auroc'), 3)} & {_fmt(r.get('token_reduction_pct'), 1)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def generate_tab_pagemantle(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate tab_pagemantle.tex (Exp 2: Deepnote PageMantle Official + PageMantle-Ext)."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Enterprise proactive performance on \\textbf{PageMantle} (\\texttt{pm\\_official\\_8} + \\texttt{pm\\_eval\\_32}) graded by Blinded Dual-Judge across Deepnote's 4 rubric axes (0--10).}",
        "\\label{tab:pagemantle}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Overall} $\\uparrow$ & \\textbf{Int/Sil} & \\textbf{Content} & \\textbf{Clarity} & \\textbf{Audience} & \\textbf{Sil Acc} & \\textbf{Timeout} & \\textbf{\\$/Task} \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r.get('method', ''))} & {_fmt(r.get('overall_score'), 2)} & "
            f"{_fmt(r.get('intervention_or_silence'), 2)} & {_fmt(r.get('content_and_evidence'), 2)} & "
            f"{_fmt(r.get('clarity_and_tone'), 2)} & {_fmt(r.get('audience_and_location'), 2)} & "
            f"{_fmt(100.0 * float(r.get('silence_accuracy', 0.0)), 1)}\\% & "
            f"{_fmt(100.0 * float(r.get('timeout_rate', 0.0)), 1)}\\% & "
            f"{_fmt(r.get('cost_per_task_usd'), 3)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def generate_tab_online_evolution(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate tab_online_evolution.tex (Exp 3: Online DB-CSE vs Reflexion / AWM / No-Escrow)."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Online non-parametric evolution across 3 sequential waves (\\texttt{pm\\_online\\_stream\\_24}) and strictly held-out \\texttt{pm\\_eval\\_32}. DB-CSE improves held-out score under strict $O(1)$ prompt size.}",
        "\\label{tab:online_evolution}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Online Updater} & \\textbf{Wave 1} & \\textbf{Wave 2} & \\textbf{Wave 3} & \\textbf{Held-Out} $\\uparrow$ & \\textbf{$\\Delta$ vs v0} & \\textbf{Prompt Ratio} & \\textbf{Escrow Blocked} \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r.get('method', ''))} & {_fmt(r.get('wave_1'), 2)} & "
            f"{_fmt(r.get('wave_2'), 2)} & {_fmt(r.get('wave_3'), 2)} & "
            f"{_fmt(r.get('heldout_score'), 2)} & {r.get('delta_vs_v0', '--')} & "
            f"{_fmt(r.get('prompt_token_ratio', 1.0), 2)}$\\times$ & {r.get('escrow_blocked', '--')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def generate_tab_ablations(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate tab_ablations.tex (Systematic ablation study across When2Speak & PageMantle)."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Systematic ablation study across When2Speak-Main (validation-calibrated per variant) and held-out PageMantle-Hard (\\texttt{pm\\_hard\\_eval\\_32}).}",
        "\\label{tab:ablations}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Variant} & \\textbf{W2S Macro F1} & \\textbf{W2S FIR} & \\textbf{PM Overall} & \\textbf{PM Audience} & \\textbf{Tok Red. (\\%)} \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(r.get('variant', ''))} & {_fmt(r.get('w2s_macro_f1'), 3)} & "
            f"{_fmt(r.get('w2s_fir'), 3)} & {_fmt(r.get('pm_overall'), 2)} & "
            f"{_fmt(r.get('pm_audience'), 2)} & {_fmt(r.get('token_reduction_pct'), 1)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def generate_tab_utterance_quality(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate tab_utterance_quality.tex evaluating When2Speak response quality against gold references."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Utterance quality on stratified \\textbf{When2Speak} gold \\texttt{SPEAK} turns graded blindly by cross-family Dual-Judge (0--10) against human reference responses.}",
        "\\label{tab:utterance_quality}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Backbone} & \\textbf{Overall} $\\uparrow$ & \\textbf{Helpfulness} & \\textbf{Conciseness} & \\textbf{Appropriateness} & \\textbf{Avg Words} & \\textbf{Perm. $p$} \\\\",
        "\\midrule",
    ]
    for r in rows:
        pval = r.get("perm_p")
        p_str = "--" if pval is None else ("$<$0.001" if float(pval) < 0.001 else _fmt(pval, 3))
        lines.append(
            f"{_tex_escape(r.get('method', ''))} & {_tex_escape(r.get('model', ''))} & "
            f"\\textbf{{{_fmt(r.get('overall_score'), 2)}}} & {_fmt(r.get('helpfulness'), 2)} & "
            f"{_fmt(r.get('conciseness'), 2)} & {_fmt(r.get('appropriateness'), 2)} & "
            f"{_fmt(r.get('avg_words'), 1)} & {p_str} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 2. Vector PDF Figure Generators (5 Publication Figures)
# ---------------------------------------------------------------------------


def plot_fir_mir_pareto(
    pareto_points: Sequence[dict[str, Any]],
    reference_points: Sequence[dict[str, Any]] | None,
    out_path: str | Path,
) -> Path:
    """Generate fig_fir_mir_pareto.pdf plotting continuous DIAL-Skill lambda frontier vs discrete baselines."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    pts = sorted(pareto_points, key=lambda d: float(d["fir"]))
    firs = [float(d["fir"]) for d in pts]
    mirs = [float(d["mir"]) for d in pts]
    ax.plot(firs, mirs, "-o", color="#1f77b4", linewidth=2.2, markersize=6, label="DIAL-Skill ($\\lambda$ sweep)")
    for d in pts:
        ax.annotate(
            f"$\\lambda$={d['lambda']}",
            (float(d["fir"]), float(d["mir"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    markers = ["s", "^", "D", "X", "P"]
    for i, ref in enumerate(reference_points or []):
        ax.scatter(
            [float(ref["fir"])],
            [float(ref["mir"])],
            marker=markers[i % len(markers)],
            s=65,
            label=str(ref["label"]),
            zorder=5,
        )

    ax.set_xlabel("False Interruption Rate (FIR) $\\downarrow$")
    ax.set_ylabel("Missed Intervention Rate (MIR) $\\downarrow$")
    ax.set_title("When2Speak FIR--MIR Pareto Frontier")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p


def plot_online_learning_curve(
    series: dict[str, Sequence[float]],
    out_path: str | Path,
    x_labels: Sequence[str] = ("Wave 1", "Wave 2", "Wave 3", "Held-Out (40)"),
) -> Path:
    """Generate fig_online_learning_curve.pdf across Waves 1-3 and Held-Out pm_eval_32."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    xs = np.arange(len(x_labels))
    styles = ["o-", "s--", "^-.", "d:", "v--"]
    for i, (label, vals) in enumerate(series.items()):
        lw = 2.5 if "DB-CSE Full" in label or "Ours" in label else 1.7
        ax.plot(xs[: len(vals)], vals, styles[i % len(styles)], linewidth=lw, markersize=6, label=label)

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Overall Rubric Score (0--10) $\\uparrow$")
    ax.set_title("Online Non-Parametric Skill Evolution Trajectory")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p


def plot_cost_vs_score(points: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate fig_cost_vs_score.pdf comparing Cost ($/task) vs Overall Score (0-10)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for pt in points:
        lbl = str(pt.get("label", ""))
        is_ours = ("dial-skill" in lbl.lower()) or ("dial_skill" in lbl.lower())
        ax.scatter(
            [float(pt["cost_per_task_usd"])],
            [float(pt["overall_score"])],
            s=110 if is_ours else 70,
            marker="*" if is_ours else "o",
            color="#d62728" if is_ours else "#4c72b0",
            zorder=5 if is_ours else 3,
        )
        ax.annotate(
            str(pt["label"]),
            (float(pt["cost_per_task_usd"]), float(pt["overall_score"])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            fontweight="bold" if is_ours else "normal",
        )

    ax.set_xlabel("Average Cost per Task (USD) $\\downarrow$")
    ax.set_ylabel("PageMantle Overall Score (0--10) $\\uparrow$")
    ax.set_title("Cost--Performance Scaling on PageMantle")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p


def plot_ablation_delta(
    ablations: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Generate fig_ablation_delta.pdf showing relative performance impacts across A1-A7.

    Supports both single-panel (`delta`) and publication dual-panel (`pm_delta` + `fir_delta`) layouts.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(a["label"]) for a in ablations]
    y = np.arange(len(labels))
    has_dual = any(("pm_delta" in a or "fir_delta" in a) for a in ablations)

    if has_dual:
        pm_deltas = [float(a.get("pm_delta", 0.0) or 0.0) for a in ablations]
        fir_deltas = [100.0 * float(a.get("fir_delta", 0.0) or 0.0) for a in ablations]
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10.2, max(4.0, 0.45 * len(labels))), sharey=True
        )
        pm_colors = ["#c44e52" if d < -0.05 else "#55a868" for d in pm_deltas]
        ax1.barh(y, pm_deltas, color=pm_colors, alpha=0.88)
        ax1.axvline(0.0, color="black", linewidth=0.9)
        ax1.set_yticks(y)
        ax1.set_yticklabels(labels, fontsize=8.5)
        ax1.invert_yaxis()
        ax1.set_xlabel("PageMantle Overall Score Change ($\\Delta$, higher is better)")
        ax1.set_title("Enterprise Quality Drop on PageMantle")
        ax1.grid(True, axis="x", linestyle="--", alpha=0.35)

        fir_colors = ["#c44e52" if d > 1.0 else "#55a868" for d in fir_deltas]
        ax2.barh(y, fir_deltas, color=fir_colors, alpha=0.88)
        ax2.axvline(0.0, color="black", linewidth=0.9)
        ax2.set_xlabel("When2Speak FIR Change ($\\Delta$ pp, lower is better)")
        ax2.set_title("False Interruption Explosion on When2Speak")
        ax2.grid(True, axis="x", linestyle="--", alpha=0.35)

        fig.suptitle("7-Point Ablation Impact vs. Full DIAL-Skill", fontsize=11)
        fig.tight_layout()
        fig.savefig(p, format="pdf")
        plt.close(fig)
        return p

    deltas = [float(a["delta"]) for a in ablations]
    colors = ["#c44e52" if d < 0 else "#55a868" for d in deltas]

    fig, ax = plt.subplots(figsize=(6.2, max(3.6, 0.45 * len(labels))))
    ax.barh(y, deltas, color=colors, alpha=0.88)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Performance Change vs. Full DIAL-Skill ($\\Delta$)")
    ax.set_title("7-Point Ablation Impact Analysis")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p


def plot_prompt_length_o1(
    steps: Sequence[int],
    dbcse_tokens: Sequence[int],
    reflexion_tokens: Sequence[int],
    awm_tokens: Sequence[int] | None,
    out_path: str | Path,
) -> Path:
    """Generate fig_prompt_length_o1.pdf illustrating bounded O(1) prompt capacity vs Reflexion O(T) growth."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(steps, dbcse_tokens, "-o", color="#1f77b4", linewidth=2.4, label="DB-CSE (Ours, $O(1)$ Cap)")
    ax.plot(steps, reflexion_tokens, "--s", color="#d62728", linewidth=2.0, label="Reflexion-Append ($O(T)$)")
    if awm_tokens is not None:
        ax.plot(steps, awm_tokens, "-.^", color="#ff7f0e", linewidth=1.8, label="AWM-Workflow")

    ax.set_xlabel("Online Stream Episodes Processed ($t$)")
    ax.set_ylabel("Skill / Memory Prompt Tokens")
    ax.set_title("Prompt Token Stability Under Online Adaptation")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True)
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 3. Qualitative Paper Case Study Auto-Extractor (Exp 6)
# ---------------------------------------------------------------------------


def extract_paper_case_studies(
    *,
    episodes: Sequence[EpisodeResult],
    tasks: Sequence[PageMantleTask],
    evolution_log: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Auto-extract 3 high-impact paper case studies: Silence Trap, High-U_t De-escalation, DB-CSE Transfer."""
    task_map = {t.task_id: t for t in tasks}

    silence_trap_case = None
    deescalation_case = None

    for ep in episodes:
        t = task_map.get(ep.sample_id)
        if t is None:
            continue
        if (
            silence_trap_case is None
            and t.expected_action == "SILENT"
            and ep.gate.final_tier == InterventionTier.SILENT
        ):
            silence_trap_case = {
                "case_type": "1_silence_trap_success",
                "task_id": t.task_id,
                "category": t.category,
                "trap_type": t.trap_type,
                "advocate_score": ep.gate.advocate_score,
                "skeptic_score": ep.gate.skeptic_score,
                "margin_delta": ep.gate.margin_delta,
                "skeptic_rationale": ep.gate.skeptic.disruption_risk if ep.gate.skeptic else "short-circuited",
                "matched_vetos": ep.gate.skeptic.matched_veto_conditions if ep.gate.skeptic else [],
            }

        if (
            deescalation_case is None
            and ep.gate.final_tier != InterventionTier.SILENT
            and ep.gate.final_tier != ep.gate.raw_tier
        ):
            deescalation_case = {
                "case_type": "2_high_ut_deescalation",
                "task_id": t.task_id,
                "category": t.category,
                "raw_tier": ep.gate.raw_tier.value,
                "final_tier": ep.gate.final_tier.value,
                "ideal_tier": t.ideal_tier.value,
                "disagreement_u": ep.gate.disagreement_u,
                "deescalation_reasons": ep.gate.deescalation_reasons,
                "response_preview": (ep.response_text or "")[:240],
            }

    dbcse_transfer_case = None
    for entry in evolution_log or []:
        if entry.get("promoted") and entry.get("patch_proposed"):
            dbcse_transfer_case = {
                "case_type": "3_dbcse_learned_boundary_transfer",
                "source_episode": entry.get("sample_id"),
                "skill_name": entry.get("skill_name"),
                "root_cause": entry.get("root_cause"),
                "patch_proposed": entry.get("patch_proposed"),
                "escrow_details": entry.get("escrow_details"),
            }
            break

    return {
        "silence_trap_success": silence_trap_case,
        "high_ut_deescalation": deescalation_case,
        "dbcse_boundary_transfer": dbcse_transfer_case,
    }


def save_case_studies(case_studies: dict[str, Any], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(case_studies, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 4. Incremental Cross-Model Generalization Tables & Figures (Notebook 07)
# ---------------------------------------------------------------------------


def generate_tab_cross_model_when2speak(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate tab_cross_model_when2speak.tex comparing Direct Chat vs DIAL-Skill across model families."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-model generalization on \\textbf{When2Speak} across OpenAI generations and open-weight/Chinese model families (zero weight updates; validation-calibrated on $\\le 200$ turns).}",
        "\\label{tab:cross_model_when2speak}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Model} & \\textbf{Direct F1} & \\textbf{DIAL F1} $\\uparrow$ & \\textbf{$\\Delta$ F1} & \\textbf{Direct FIR} & \\textbf{DIAL FIR} $\\downarrow$ & \\textbf{DIAL AUROC} & \\textbf{Holm $p$} \\\\",
        "\\midrule",
    ]
    for r in rows:
        d_f1 = r.get("direct_macro_f1")
        o_f1 = r.get("dial_macro_f1")
        delta = (float(o_f1) - float(d_f1)) if (d_f1 is not None and o_f1 is not None) else None
        delta_str = f"{delta:+.3f}" if delta is not None else "--"
        pval = r.get("holm_bootstrap_p", r.get("bootstrap_p", r.get("mcnemar_p")))
        p_str = "$<$0.001" if (isinstance(pval, (int, float)) and pval < 0.001) else _fmt(pval, 3)
        lines.append(
            f"{_tex_escape(r.get('family', ''))} & {_tex_escape(r.get('model', ''))} & "
            f"{_fmt(d_f1, 3)} & \\textbf{{{_fmt(o_f1, 3)}}} & {delta_str} & "
            f"{_fmt(r.get('direct_fir'), 3)} & {_fmt(r.get('dial_fir'), 3)} & "
            f"{_fmt(r.get('dial_auroc'), 3)} & {p_str} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def generate_tab_cross_model_pagemantle(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate tab_cross_model_pagemantle.tex comparing Direct Chat vs DIAL-Skill on PageMantle across families."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-model generalization on \\textbf{PageMantle} (0--10 Blinded Dual-Judge) across OpenAI and open-weight/Chinese backbones.}",
        "\\label{tab:cross_model_pagemantle}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Model} & \\textbf{Direct Overall} & \\textbf{DIAL Overall} $\\uparrow$ & \\textbf{$\\Delta$} & \\textbf{Sil Acc} & \\textbf{Act Rec} & \\textbf{Audience} & \\textbf{\\$/Task} \\\\",
        "\\midrule",
    ]
    for r in rows:
        d_ov = r.get("direct_overall")
        o_ov = r.get("dial_overall")
        delta = (float(o_ov) - float(d_ov)) if (d_ov is not None and o_ov is not None) else None
        delta_str = f"{delta:+.2f}" if delta is not None else "--"
        sil_val = r.get("dial_silence_accuracy")
        act_val = r.get("dial_action_recall")
        sil_str = f"{100.0 * float(sil_val):.1f}\\%" if sil_val is not None else "--"
        act_str = f"{100.0 * float(act_val):.1f}\\%" if act_val is not None else "--"
        lines.append(
            f"{_tex_escape(r.get('family', ''))} & {_tex_escape(r.get('model', ''))} & "
            f"{_fmt(d_ov, 2)} & \\textbf{{{_fmt(o_ov, 2)}}} & {delta_str} & "
            f"{sil_str} & {act_str} & {_fmt(r.get('dial_audience'), 2)} & "
            f"{_fmt(r.get('dial_cost_per_task_usd'), 4)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def plot_cross_model_w2s_gain(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate fig_cross_model_w2s_gain.pdf (Direct vs DIAL-Skill Macro F1 and FIR across model families)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(r["model"]) for r in rows]
    direct_f1 = [float(r["direct_macro_f1"]) if r.get("direct_macro_f1") is not None else np.nan for r in rows]
    dial_f1 = [float(r["dial_macro_f1"]) if r.get("dial_macro_f1") is not None else np.nan for r in rows]
    direct_fir = [float(r["direct_fir"]) if r.get("direct_fir") is not None else np.nan for r in rows]
    dial_fir = [float(r["dial_fir"]) if r.get("dial_fir") is not None else np.nan for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(9.5, 1.15 * len(labels)), 4.1))

    ax1.bar(x - w / 2, direct_f1, width=w, color="#9aa5b1", label="Direct Chat (B0)")
    ax1.bar(x + w / 2, dial_f1, width=w, color="#1f77b4", label="DIAL-Skill (Ours)")
    ax1.axhline(0.567, color="#d62728", linestyle="--", linewidth=1.2, label="GPT-4o Zero-Shot Ref (0.567)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    ax1.set_ylabel("Macro F1 $\\uparrow$")
    ax1.set_title("When2Speak Macro F1 Across Model Families")
    ax1.set_ylim(0.0, 0.90)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    ax2.bar(x - w / 2, direct_fir, width=w, color="#e07a5f", label="Direct Chat FIR")
    ax2.bar(x + w / 2, dial_fir, width=w, color="#2a9d8f", label="DIAL-Skill FIR (Ours)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    ax2.set_ylabel("False Interruption Rate (FIR) $\\downarrow$")
    ax2.set_title("Interruption Restraint Across Model Families")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p


def plot_cross_model_pagemantle_gain(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Generate fig_cross_model_pagemantle_gain.pdf (Direct vs DIAL-Skill PageMantle score across families)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(r["model"]) for r in rows]
    direct_ov = [float(r.get("direct_overall", 0.0) or 0.0) for r in rows]
    dial_ov = [float(r.get("dial_overall", 0.0) or 0.0) for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(6.4, 0.95 * len(labels)), 4.2))
    ax.bar(x - w / 2, direct_ov, width=w, color="#9aa5b1", label="Direct Proactive Agent (B0)")
    ax.bar(x + w / 2, dial_ov, width=w, color="#2b6cb0", label="DIAL-Skill (Ours)")
    for i, (d, o) in enumerate(zip(direct_ov, dial_ov)):
        if o > 0:
            ax.annotate(f"{o:.2f}", (x[i] + w / 2, o), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, fontweight="bold")
    ax.axhline(6.38, color="#e67e22", linestyle=":", linewidth=1.2, label="Deepnote Ref: GPT-6 Luna (6.38)")
    ax.axhline(8.65, color="#27ae60", linestyle="--", linewidth=1.2, label="Deepnote Ref: GPT-6.1 Sol (8.65)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("PageMantle Overall Score (0--10) $\\uparrow$")
    ax.set_title("Cross-Model Generalization on Enterprise PageMantle")
    ax.set_ylim(0.0, 10.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p


def plot_cross_family_judge_agreement(
    primary_scores: Sequence[float],
    secondary_scores: Sequence[float],
    primary_label: str,
    secondary_label: str,
    spearman_rho: float,
    out_path: str | Path,
) -> Path:
    """Generate fig_cross_family_judge_agreement.pdf verifying absence of single-vendor judge bias."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    xs = np.asarray(primary_scores, dtype=float)
    ys = np.asarray(secondary_scores, dtype=float)
    ax.scatter(xs, ys, s=55, alpha=0.80, color="#1f77b4", edgecolors="white", linewidth=0.6)
    ax.plot([0, 10], [0, 10], "--", color="#666666", linewidth=1.1, label="Identity ($y = x$)")
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.3, 10.3)
    ax.set_xlabel(f"Primary Judge: {primary_label} (0--10)")
    ax.set_ylabel(f"Cross-Family Judge: {secondary_label} (0--10)")
    ax.set_title(f"Cross-Family Blinded Judge Agreement ($\\rho = {spearman_rho:.3f}$)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(p, format="pdf")
    plt.close(fig)
    return p
