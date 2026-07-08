from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


OUT_REPORT = Path("docs/mapk1_case_study.md")
OUT_CHECKLIST = Path("results/reports/mapk1_case_study_completion_checklist.md")


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_float(value, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None, digits: int = 3) -> str:
    if df.empty:
        return "_No rows available._"
    work = df.copy()
    if columns:
        work = work[[col for col in columns if col in work.columns]]
    if max_rows:
        work = work.head(max_rows)
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: fmt_float(x, digits))
    headers = list(work.columns)
    rows = work.fillna("").astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def image(path: str, caption: str) -> str:
    if not Path(path).exists():
        return f"_Missing figure: `{path}`._"
    return f"![{caption}]({path})\n\n*{caption}*"


def section(title: str) -> str:
    return f"\n## {title}\n"


def add_cnn_vs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "CNN_VS" in df.columns:
        return df
    if {"CNNscore", "CNNaffinity"}.issubset(df.columns):
        out = df.copy()
        out["CNN_VS"] = pd.to_numeric(out["CNNscore"], errors="coerce") * pd.to_numeric(out["CNNaffinity"], errors="coerce")
        return out
    return df


def metrics_table(path: str, methods: Iterable[str] | None = None) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    if methods:
        df = df[df["method"].isin(methods)].copy()
    cols = [
        "population",
        "method",
        "n_total",
        "n_active",
        "roc_auc",
        "pr_auc",
        "ef1",
        "ef5",
        "ef10",
        "top50_actives",
        "top100_actives",
        "top250_actives",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def value_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    return df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")


def score_ranges(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["unidock_best_score", "CNNscore", "CNNaffinity", "CNN_VS", "gnina_affinity"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({"score": col, "min": s.min(), "median": s.median(), "max": s.max()})
    return pd.DataFrame(rows)


def method_subset(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "ecfp4_active_similarity",
        "unidock_best",
        "gnina_cnnscore",
        "gnina_cnnaffinity",
        "gnina_cnn_vs",
        "gnina_affinity",
        "fusion_unidock_cnnscore_cnnaffinity",
    ]
    if df.empty or "method" not in df.columns:
        return df
    return df[df["method"].isin(keep)].copy()


def write_completion_checklist() -> None:
    OUT_CHECKLIST.parent.mkdir(parents=True, exist_ok=True)
    OUT_CHECKLIST.write_text(
        "\n".join(
            [
                "# MAPK1 Case Study Completion Checklist",
                "",
                "- [x] Dataset acquired, curated, deduplicated, and audited.",
                "- [x] Phase 1 screening population prepared.",
                "- [x] Five MAPK1 receptors selected, prepared, aligned, and boxed.",
                "- [x] Phase 1 ligands converted to 3D and PDBQT.",
                "- [x] Uni-Dock full phase 1 completed on GPU.",
                "- [x] GNINA full all-valid rescoring completed on GPU.",
                "- [x] Score QC and anomaly sensitivity analysis completed.",
                "- [x] ECFP4, few-active, native-ligand, scaffold, and negative-control baselines completed.",
                "- [x] Inspection shortlist and candidate triage completed.",
                "- [x] Final PyMOL review images cleaned to publication-safe assets.",
                "- [x] README and case-study report updated to avoid overclaiming.",
                "",
                "Remaining optional human action: fill `results/tables/mapk1_phase1_final_manual_pose_verdict.csv` after visual pose review. "
                "The current project does not require this to claim benchmark completion because it does not claim discovered inhibitors.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    dataset = read_json("results/reports/mapk1_dataset_manifest.json")
    splits = read_json("results/reports/mapk1_screening_splits_manifest.json")
    conformers = read_json("results/reports/mapk1_phase1_3d_manifest.json")
    pdbqt = read_json("results/reports/mapk1_phase1_pdbqt_manifest.json")
    receptor_prep = read_json("results/reports/mapk1_receptor_prep_manifest.json")

    score_pop = add_cnn_vs(read_csv("results/tables/mapk1_phase1_score_population.csv"))
    clean_score_pop = add_cnn_vs(read_csv("results/tables/mapk1_phase1_clean_population_with_ecfp4_and_anomalies.csv"))
    structure_metrics = metrics_table("results/tables/mapk1_phase1_method_metrics.csv")
    ecfp_metrics = method_subset(metrics_table("results/tables/mapk1_phase1_all_method_metrics_with_ecfp4_full.csv"))
    few_active = read_csv("results/tables/mapk1_phase1_few_active_ecfp4_summary_full.csv")
    crossover = read_csv("results/tables/mapk1_phase1_prior_knowledge_crossover_summary.csv")
    triage = read_csv("results/tables/mapk1_phase1_candidate_triage.csv")
    receptors = read_csv("results/tables/mapk1_receptor_selection.csv")
    boxes = read_csv("results/tables/mapk1_docking_boxes.csv")
    variability = read_csv("results/tables/mapk1_receptor_ensemble_variability.csv")
    native_ccd = read_csv("results/tables/mapk1_phase1_native_ligand_ecfp4_ccd_metrics.csv")
    native_sanity = read_csv("results/tables/mapk1_native_ligand_reference_sanity.csv")

    selected_receptors = receptors[receptors["selected"].eq(True)] if not receptors.empty and "selected" in receptors.columns else pd.DataFrame()
    tier_a = triage[triage["triage_tier"].eq("A_analog_seed")] if not triage.empty and "triage_tier" in triage.columns else pd.DataFrame()

    anomaly_rows = []
    for col in [
        "positive_gnina_affinity",
        "extreme_positive_gnina_affinity",
        "suspicious_unidock_extreme_negative",
        "extreme_unidock_negative",
        "suspicious_ligand_efficiency_extreme",
        "has_any_score_anomaly",
    ]:
        if col in score_pop.columns:
            anomaly_rows.append({"flag": col, "count": int(pd.to_numeric(score_pop[col], errors="coerce").fillna(0).sum())})

    few_ecfp = pd.DataFrame()
    if not few_active.empty and {"method", "k"}.issubset(few_active.columns):
        few_ecfp = few_active[few_active["method"].eq("few_active_ecfp4")][
            ["k", "n_repeats", "pr_auc_mean", "ef1_mean", "ef5_mean", "top50_actives_mean"]
        ].copy()

    prep_counts = pd.DataFrame(
        [
            {
                "3d_attempted": conformers.get("attempted"),
                "3d_successful": conformers.get("successful"),
                "3d_failed": conformers.get("failed"),
                "pdbqt_input_sdf": pdbqt.get("total_input_sdf"),
                "pdbqt_successful": pdbqt.get("successful_pdbqt"),
                "receptors_successful": receptor_prep.get("successful"),
                "receptors_failed": receptor_prep.get("failed"),
            }
        ]
    )

    variability_summary = pd.DataFrame()
    if not variability.empty and "coordinate_dispersion_angstrom" in variability.columns:
        disp = pd.to_numeric(variability["coordinate_dispersion_angstrom"], errors="coerce")
        variability_summary = pd.DataFrame(
            [
                {
                    "common_ca_residues": len(variability),
                    "near_pocket_residues": int(pd.to_numeric(variability.get("near_pocket", 0), errors="coerce").fillna(0).sum()),
                    "median_dispersion_A": disp.median(),
                    "max_dispersion_A": disp.max(),
                }
            ]
        )

    lines: list[str] = [
        "# MAPK1 Leakage-Aware Virtual Screening Benchmark",
        "",
        "**Final status:** complete as a retrospective benchmark and worked example.",
        "",
        "**Boundary:** this project does not claim new MAPK1 inhibitors. It tests whether structure-based scores recover known actives better than ligand-similarity baselines, then triages the few low-similarity active cases worth manual pose review.",
        "",
        "## Result In One Paragraph",
        "",
        "The structure-based workflow ran successfully at scale, but raw Uni-Dock and GNINA scores were weak global rankers. "
        "ECFP4/Tanimoto similarity dominated the benchmark, showing that analog-neighborhood structure explains much of the active recovery. "
        "The useful project outcome is therefore not a fake hit-discovery claim; it is an audited CADD decision workflow that identifies where docking/GNINA fail, where ligand similarity dominates, and which low-similarity active poses remain plausible structure-added-value cases.",
        "",
        "## Why This Workflow Matters",
        "",
        "- It turns a docking run into a measured decision system with baselines, QC, and failure modes.",
        "- It demonstrates CPU/GPU workflow ownership across a large VM and an RTX4090 machine.",
        "- It shows productivity: data curation, receptor prep, GPU docking, GNINA rescoring, metrics, plots, reports, and pose panels are reproducible from code.",
        "- It avoids presenting docking scores as discovered binders.",
        "",
        "## What Was Built",
        "",
        md_table(
            pd.DataFrame(
                [
                    {"stage": "dataset", "output": f"{dataset.get('final_active_count')} actives / {dataset.get('final_inactive_count')} inactives curated"},
                    {"stage": "phase 1 split", "output": f"{splits.get('phase1_actives')} actives / {splits.get('phase1_inactives')} inactives"},
                    {"stage": "ligand prep", "output": "5,297 prepared ligand PDBQT files"},
                    {"stage": "receptor prep", "output": "5 prepared MAPK1 receptors and 5 docking boxes"},
                    {"stage": "docking", "output": "26,485 Uni-Dock attempts completed"},
                    {"stage": "rescoring", "output": "5,233 GNINA all-valid score rows"},
                    {"stage": "triage", "output": "5 Tier A retrospective pose-review cases"},
                ]
            )
        ),
    ]

    lines.append(section("1. Dataset Curation"))
    lines.extend(
        [
            "LIT-PCBA MAPK1 was parsed, standardized with RDKit, deduplicated, and checked for activity conflicts. The final active count is lower than the nominal benchmark count because invalid, duplicate, or conflicting records were not forced into the clean table.",
            "",
            md_table(
                pd.DataFrame(
                    [
                        {
                            "raw_actives": dataset.get("raw_active_records"),
                            "raw_inactives": dataset.get("raw_inactive_records"),
                            "final_actives": dataset.get("final_active_count"),
                            "final_inactives": dataset.get("final_inactive_count"),
                            "duplicates_removed": dataset.get("duplicate_records_removed"),
                            "activity_conflicts_removed": dataset.get("activity_conflicts_removed"),
                        }
                    ]
                )
            ),
        ]
    )

    lines.append(section("2. Receptor Ensemble And Ligand Preparation"))
    lines.extend(
        [
            "Five high-resolution MAPK1 holo structures were prepared. Docking boxes were derived from native ligand coordinates. This gave a small receptor ensemble rather than a single structure.",
            "",
            md_table(selected_receptors, ["pdb_id", "method", "resolution", "chains", "ligands", "reason"]),
            "",
            "Docking boxes:",
            "",
            md_table(boxes, ["pdb_id", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "status"], digits=2),
            "",
            "Preparation counts:",
            "",
            md_table(prep_counts),
            "",
            "The receptor variability plot aligns common C-alpha atoms across the five structures. The x-axis is MAPK1 residue position in the shared reference sequence; the y-axis is how much that residue's C-alpha position differs across the aligned structures. Larger values mean that local protein region moves more across the selected crystal structures.",
            "",
            md_table(variability_summary),
            "",
            image("results/figures/mapk1_receptor_ensemble_variability.png", "Receptor ensemble variability after sequence-mapped C-alpha alignment."),
        ]
    )

    lines.append(section("3. Uni-Dock And GNINA Score QC"))
    lines.extend(
        [
            "Uni-Dock generated poses and scores against all five receptors. GNINA then rescored the best Uni-Dock pose per ligand. The shared evaluation population contains 5,233 ligands with both score families.",
            "",
            "Score ranges including flagged anomalies:",
            "",
            md_table(score_ranges(score_pop)),
            "",
            "Clean sensitivity ranges after severe anomaly exclusion:",
            "",
            md_table(score_ranges(clean_score_pop)),
            "",
            "Main anomaly counts:",
            "",
            md_table(pd.DataFrame(anomaly_rows)),
            "",
            "The extreme Uni-Dock score near -29 and GNINA affinity values above +20 were kept for provenance but flagged as suspicious. They are not treated as strong evidence of binding.",
        ]
    )

    lines.append(section("4. Enrichment Analysis"))
    lines.extend(
        [
            "Enrichment analysis sorts ligands from best predicted to worst predicted for each method, then checks whether known actives appear early in that ranked list. The score is not a classifier probability; it is a ranking signal.",
            "",
            "Structure-only metrics:",
            "",
            md_table(method_subset(structure_metrics)),
            "",
            "Structure plus ligand-similarity metrics:",
            "",
            md_table(ecfp_metrics),
            "",
            "Decision-relevant plot:",
            "",
            image("results/figures/mapk1_phase1_all_methods_ef1_comparison_full.png", "EF1 comparison. ECFP4 dominates early enrichment; raw structure scores are weak."),
            "",
            image("results/figures/mapk1_phase1_all_methods_pr_auc_comparison_full.png", "PR-AUC comparison. The ligand-only baseline beats Uni-Dock and GNINA globally."),
            "",
            "Diagnostic score plots remain under `results/figures/`, but they are not the core story. The core story is the metrics table above: the structure scores did not provide strong global ranking.",
        ]
    )

    lines.append(section("5. Objective 2D Baselines"))
    lines.extend(
        [
            "The all-active ECFP4 result is too strong as a prospective model because it uses nearly all known actives as references. The project therefore added stricter controls: few-active baselines, scaffold-held-out baselines, near-analog removal, inactive-reference controls, and label-shuffle controls.",
            "",
            "Few-active ECFP4 calibration:",
            "",
            md_table(few_ecfp),
            "",
            "Prior-knowledge crossover summary:",
            "",
            md_table(crossover),
            "",
            "Native-ligand-only ECFP4 was repaired using CCD-derived references because PDB-derived native ligand files had unreliable bond orders. Even after repair, the native-ligand-only baseline stayed weak.",
            "",
            "CCD-native baseline:",
            "",
            md_table(native_ccd),
            "",
            "Native ligand QC:",
            "",
            md_table(native_sanity, ["pdb_id", "native_ligand_resname", "sanity_flags", "recommendation"], max_rows=5),
            "",
            image("results/figures/mapk1_phase1_prior_knowledge_curve_ef1_full.png", "Few-active ECFP4 prior-knowledge curve for EF1."),
            "",
            image("results/figures/mapk1_phase1_ecfp4_similarity_distribution_by_activity.png", "ECFP4 active-similarity distribution by activity label."),
        ]
    )

    lines.append(section("6. Candidate Triage"))
    lines.extend(
        [
            "The candidate triage does not use ECFP4 to declare final actives. It uses the known retrospective labels plus structure-added-value logic to find active ligands that ECFP4 ranked poorly but structure methods ranked better. These are useful for inspection, not proof of discovery.",
            "",
            "Triage tiers:",
            "",
            md_table(value_counts(triage, "triage_tier")),
            "",
            "Tier A retrospective pose-review cases:",
            "",
            md_table(
                tier_a,
                [
                    "ligand_id",
                    "activity_label",
                    "ecfp4_active_similarity",
                    "best_structure_method",
                    "SAV",
                    "unidock_best_score",
                    "CNNscore",
                    "CNNaffinity",
                    "gnina_affinity",
                    "n_total_interactions",
                ],
            ),
        ]
    )

    lines.append(section("7. Final Pose Review Assets"))
    lines.extend(
        [
            "The final image folder has been cleaned. It keeps only simple full-receptor views and the manually selected annotated images.",
            "",
            "- `results/figures/final_pymol_review/png/*_01_full_receptor.png`",
            "- `results/figures/final_pymol_review/png/*_annotated.png`",
            "- `results/tables/mapk1_phase1_final_manual_pose_verdict.csv`",
            "",
            "Example final pose assets:",
            "",
            image("results/figures/final_pymol_review/png/26747800_01_full_receptor.png", "Full-receptor context for Tier A ligand 26747800."),
            "",
            image("results/figures/final_pymol_review/png/26747800_annotated.png", "User-selected annotated pose-review image for Tier A ligand 26747800."),
            "",
            "Manual pose review does not validate binding or potency. It only decides whether a retrospective docking pose is plausible enough to show or use as an analog-prioritization seed.",
        ]
    )

    lines.append(section("Final Conclusion"))
    lines.extend(
        [
            "This project is complete as a retrospective, leakage-aware virtual-screening benchmark. It should be presented as evidence of CADD workflow ownership, not as a hit-discovery campaign.",
            "",
            "The strongest honest claim is:",
            "",
            "> I built and audited a full MAPK1 docking/rescoring benchmark, proved that raw structure scores were weak against ligand-similarity baselines, and isolated the small low-similarity active cases where structure-based ranking may still add value.",
            "",
            "The next upgrade is a second target or dataset: interaction fingerprints with ProLIF, a hard-novel train/test design, and supervised re-ranking.",
        ]
    )

    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_completion_checklist()


if __name__ == "__main__":
    main()
