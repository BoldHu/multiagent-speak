"""Milestone 6 tests: LaTeX tables, matplotlib vector PDF figures, case study extractor, notebook validation & secret safety."""

from __future__ import annotations

from pathlib import Path

from dial_skill.config import find_repo_root
from dial_skill.eval.report import (
    extract_paper_case_studies,
    generate_tab_ablations,
    generate_tab_online_evolution,
    generate_tab_pagemantle,
    generate_tab_utterance_quality,
    generate_tab_when2speak,
    plot_ablation_delta,
    plot_cost_vs_score,
    plot_fir_mir_pareto,
    plot_online_learning_curve,
    plot_prompt_length_o1,
)
from dial_skill.schemas import (
    ChatMessage,
    EpisodeResult,
    GateDecision,
    InterventionTier,
    PageMantleTask,
    SkepticOutput,
)
from scripts.validate_notebooks import EXPECTED_NOTEBOOKS, validate_all_notebooks


def test_latex_table_generators(tmp_path: Path) -> None:
    t1 = generate_tab_when2speak(
        [
            {"method": "B0 Direct Chat", "model": "Luna", "macro_f1": 0.56, "speak_f1": 0.34, "fir": 0.22, "mir": 0.45, "auroc": 0.65, "token_reduction_pct": 0.0},
            {"method": "DIAL-Skill (Ours)", "model": "Luna", "macro_f1": 0.74, "speak_f1": 0.58, "fir": 0.06, "mir": 0.26, "auroc": 0.89, "token_reduction_pct": 54.2},
        ],
        tmp_path / "tab_when2speak.tex",
    )
    t2 = generate_tab_pagemantle(
        [
            {"method": "DIAL-Skill-Evolved", "overall_score": 8.2, "intervention_or_silence": 9.1, "content_and_evidence": 7.9, "clarity_and_tone": 8.4, "audience_and_location": 8.6, "silence_accuracy": 0.90, "timeout_rate": 0.0, "cost_per_task_usd": 0.042},
        ],
        tmp_path / "tab_pagemantle.tex",
    )
    t3 = generate_tab_online_evolution(
        [
            {"method": "Static-v0", "wave_1": 7.1, "wave_2": 7.0, "wave_3": 7.2, "heldout_score": 7.1, "delta_vs_v0": "+0.00", "prompt_token_ratio": 1.0, "escrow_blocked": 0},
            {"method": "DB-CSE Full", "wave_1": 7.2, "wave_2": 7.6, "wave_3": 8.1, "heldout_score": 7.9, "delta_vs_v0": "+0.80", "prompt_token_ratio": 1.08, "escrow_blocked": 2},
        ],
        tmp_path / "tab_online_evolution.tex",
    )
    t4 = generate_tab_ablations(
        [
            {"variant": "Full DIAL-Skill", "w2s_macro_f1": 0.74, "w2s_fir": 0.06, "pm_overall": 8.2, "pm_audience": 8.6, "token_reduction_pct": 54.2},
            {"variant": "A3 Asymmetric Evidence", "w2s_macro_f1": 0.66, "w2s_fir": 0.13, "pm_overall": 7.4, "pm_audience": 7.3, "token_reduction_pct": 53.0},
        ],
        tmp_path / "tab_ablations.tex",
    )

    t5 = generate_tab_utterance_quality(
        [
            {"method": "B0 Direct", "model": "gpt-4.1-mini", "overall_score": 6.4, "helpfulness": 6.8, "conciseness": 5.6, "appropriateness": 6.5, "avg_words": 74.2, "perm_p": 0.0004},
            {"method": "DIAL-Collab (Audited)", "model": "gpt-4.1-mini", "overall_score": 8.3, "helpfulness": 8.4, "conciseness": 8.5, "appropriateness": 8.2, "avg_words": 38.5, "perm_p": None},
        ],
        tmp_path / "tab_utterance_quality.tex",
    )

    for p in (t1, t2, t3, t4, t5):
        txt = p.read_text(encoding="utf-8")
        assert "\\begin{tabular}" in txt
        assert "\\toprule" in txt and "\\bottomrule" in txt


def test_vector_pdf_figure_generators(tmp_path: Path) -> None:
    f1 = plot_fir_mir_pareto(
        [{"lambda": 0.5, "fir": 0.14, "mir": 0.16}, {"lambda": 1.0, "fir": 0.06, "mir": 0.25}, {"lambda": 1.5, "fir": 0.02, "mir": 0.42}],
        [{"label": "When2Speak SFT", "fir": 0.04, "mir": 0.50}],
        tmp_path / "fig_fir_mir_pareto.pdf",
    )
    f2 = plot_online_learning_curve(
        {"Static-v0": [7.1, 7.1, 7.1, 7.1], "Reflexion": [7.1, 7.3, 7.0, 7.2], "DB-CSE Full (Ours)": [7.2, 7.6, 8.1, 8.0]},
        tmp_path / "fig_online_learning_curve.pdf",
    )
    f3 = plot_cost_vs_score(
        [{"label": "Direct (Luna)", "cost_per_task_usd": 0.02, "overall_score": 6.4}, {"label": "DIAL-Skill (Terra)", "cost_per_task_usd": 0.05, "overall_score": 8.3}],
        tmp_path / "fig_cost_vs_score.pdf",
    )
    f4 = plot_ablation_delta(
        [{"label": "A1 w/o Jev Triage", "delta": -0.05}, {"label": "A3 Asymmetric Evidence", "delta": -0.08}, {"label": "A5 Binary Only", "delta": -0.06}],
        tmp_path / "fig_ablation_delta.pdf",
    )
    f5 = plot_prompt_length_o1(
        [0, 8, 16, 24],
        dbcse_tokens=[1200, 1240, 1280, 1295],
        reflexion_tokens=[1200, 1650, 2150, 2700],
        awm_tokens=[1200, 1450, 1750, 2000],
        out_path=tmp_path / "fig_prompt_length_o1.pdf",
    )
    for p in (f1, f2, f3, f4, f5):
        raw = p.read_bytes()
        assert raw.startswith(b"%PDF") and len(raw) > 1000


def test_case_study_extractor() -> None:
    tasks = [
        PageMantleTask(
            task_id="t_trap",
            split="pm_eval_32",
            category="check",
            messages=[ChatMessage(msg_id="m0", speaker="a", text="resolved")],
            expected_action="SILENT",
            ideal_tier=InterventionTier.SILENT,
            trap_type="already_resolved",
        ),
        PageMantleTask(
            task_id="t_deesc",
            split="pm_eval_32",
            category="guard",
            messages=[ChatMessage(msg_id="m0", speaker="a", text="share PII?")],
            expected_action="INTERVENE",
            ideal_tier=InterventionTier.DM_NUDGE,
        ),
    ]
    eps = [
        EpisodeResult(
            sample_id="t_trap",
            benchmark="pagemantle_ext",
            method="dial_skill",
            model_id="luna",
            gate=GateDecision(
                slate_hash="h1",
                selected_skill="check",
                advocate_score=0.35,
                skeptic_score=0.88,
                margin_delta=-0.53,
                disagreement_u=0.35,
                raw_tier=InterventionTier.SILENT,
                final_tier=InterventionTier.SILENT,
                binary_speak=False,
                skeptic=SkepticOutput(
                    slate_hash="h1",
                    skeptic_score=0.88,
                    disruption_risk="Already resolved.",
                    matched_veto_conditions=["A teammate already shared the answer."],
                    max_acceptable_channel=InterventionTier.SILENT,
                    cited_evidence_ids=["m0"],
                ),
            ),
        ),
        EpisodeResult(
            sample_id="t_deesc",
            benchmark="pagemantle_ext",
            method="dial_skill",
            model_id="luna",
            gate=GateDecision(
                slate_hash="h2",
                selected_skill="guard",
                advocate_score=0.92,
                skeptic_score=0.60,
                margin_delta=0.32,
                disagreement_u=0.60,
                raw_tier=InterventionTier.THREAD_BRIEF,
                final_tier=InterventionTier.DM_NUDGE,
                binary_speak=True,
                deescalation_reasons=["epistemic_disagreement_u=0.60>=0.55"],
            ),
            response_text="Quick private heads-up: Notion policy prohibits posting PII here.",
        ),
    ]
    evo_log = [
        {
            "sample_id": "pm_stream_w1_07",
            "skill_name": "check",
            "root_cause": "VETO_MISSING",
            "promoted": True,
            "patch_proposed": {"target_field": "veto_conditions", "proposed_predicate": "Defer when owner postpones check."},
            "escrow_details": {"anchor_regressions": 0},
        }
    ]
    cases = extract_paper_case_studies(episodes=eps, tasks=tasks, evolution_log=evo_log)
    assert cases["silence_trap_success"]["task_id"] == "t_trap"
    assert cases["high_ut_deescalation"]["final_tier"] == "DM_NUDGE"
    assert cases["dbcse_boundary_transfer"]["skill_name"] == "check"


def test_all_seven_notebooks_valid_and_no_secret_files_tracked() -> None:
    import ast
    import importlib
    import nbformat
    from scripts.validate_notebooks import _strip_ipython_magics

    repo_root = find_repo_root()
    nb_dir = repo_root / "notebooks"
    assert len(EXPECTED_NOTEBOOKS) == 13
    errors = validate_all_notebooks(nb_dir)
    assert errors == [], f"Notebook validation errors: {errors}"

    # Verify .gitignore blocks JEV_API_KEY.txt, OPENAI_KEY.txt, and AUTODL_KEY.txt
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "JEV_API_KEY.txt" in gitignore
    assert "OPENAI_KEY.txt" in gitignore
    assert "AUTODL_KEY.txt" in gitignore

    # Verify every `from dial_skill... import ...` symbol across Notebooks 10, 11, 12 exists!
    for nb_name in (
        "10_when2speak_collab_and_utterance_quality.ipynb",
        "11_official_deepnote_8_clean_multiseed.ipynb",
        "12_pagemantle_hard_v3_ablations_and_dbcse.ipynb",
    ):
        nb = nbformat.read(nb_dir / nb_name, as_version=4)
        checked = 0
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            tree = ast.parse(_strip_ipython_magics(cell.source))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("dial_skill"):
                    mod = importlib.import_module(node.module)
                    for alias in node.names:
                        assert hasattr(mod, alias.name), (
                            f"{nb_name} imports missing symbol {alias.name!r} from {node.module!r}"
                        )
                        checked += 1
        assert checked >= 8, f"Expected >=8 verified imports in {nb_name}, got {checked}"


def test_cross_model_tables_and_figures(tmp_path: Path) -> None:
    from dial_skill.eval.report import (
        generate_tab_cross_model_pagemantle,
        generate_tab_cross_model_when2speak,
        plot_cross_family_judge_agreement,
        plot_cross_model_pagemantle_gain,
        plot_cross_model_w2s_gain,
    )

    w2s_rows = [
        {"family": "OpenAI-4.1", "model": "gpt-4.1-mini", "direct_macro_f1": 0.40, "dial_macro_f1": 0.62, "direct_fir": 0.60, "dial_fir": 0.05, "dial_auroc": 0.69, "mcnemar_p": 0.0},
        {"family": "DeepSeek", "model": "DeepSeek-V4.1-Flash", "direct_macro_f1": 0.45, "dial_macro_f1": 0.65, "direct_fir": 0.48, "dial_fir": 0.07, "dial_auroc": 0.74, "mcnemar_p": 0.0002},
    ]
    pm_rows = [
        {"family": "OpenAI-4.1", "model": "gpt-4.1", "direct_overall": 4.90, "dial_overall": 8.61, "dial_silence_accuracy": 1.0, "dial_action_recall": 0.97, "dial_audience": 8.41, "dial_cost_per_task_usd": 0.0059},
        {"family": "Qwen", "model": "Qwen3.5-397B-A17B", "direct_overall": 6.10, "dial_overall": 8.40, "dial_silence_accuracy": 0.90, "dial_action_recall": 0.92, "dial_audience": 8.20, "dial_cost_per_task_usd": 0.0040},
    ]
    t1 = generate_tab_cross_model_when2speak(w2s_rows, tmp_path / "tab_cross_w2s.tex")
    t2 = generate_tab_cross_model_pagemantle(pm_rows, tmp_path / "tab_cross_pm.tex")
    assert "\\begin{tabular}" in t1.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in t2.read_text(encoding="utf-8")

    f1 = plot_cross_model_w2s_gain(w2s_rows, tmp_path / "fig_cross_w2s.pdf")
    f2 = plot_cross_model_pagemantle_gain(pm_rows, tmp_path / "fig_cross_pm.pdf")
    f3 = plot_cross_family_judge_agreement([8.0, 9.0, 5.0, 10.0], [7.8, 8.9, 5.4, 9.8], "gpt-4.1", "Qwen3.5-397B-A17B", 0.96, tmp_path / "fig_judge.pdf")
    for p in (f1, f2, f3):
        assert p.read_bytes().startswith(b"%PDF") and len(p.read_bytes()) > 1000
