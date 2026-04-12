from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.collection.visualization import (
    create_comparison_visualizations,
    create_visualizations,
)
from pipeline.preprocessing.cleaning import (
    CleaningStats,
    adjust_half_step_ratings,
    clean_dataset,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
CLEANED_DIR = DATA_DIR / "cleaned"
COMPARISON_DIR = FIGURES_DIR / "comparison"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source_path: Path
    cleaned_path: Path


def rating_balance_summary(df: pd.DataFrame) -> tuple[pd.Series, float, str]:
    if "rating" not in df.columns:
        return pd.Series(dtype=int), float("nan"), "cannot-evaluate"

    score = df["rating"].dropna()
    if score.empty:
        return pd.Series(dtype=int), float("nan"), "cannot-evaluate"

    bins = pd.cut(
        score,
        bins=[-np.inf, 4, 7, np.inf],
        labels=["negative", "neutral", "positive"],
        right=False,
    )
    counts = bins.value_counts().sort_index()
    ratio = counts.min() / counts.max() if counts.max() > 0 else float("nan")

    if ratio >= 0.6:
        verdict = "balanced"
    elif ratio >= 0.35:
        verdict = "moderately-imbalanced"
    else:
        verdict = "imbalanced"

    return counts, float(ratio), verdict


def build_report(
    dataset_name: str,
    source_path: Path,
    cleaned_path: Path,
    cleaned_df: pd.DataFrame,
    stats: CleaningStats,
    figure_paths: list[Path],
) -> str:
    missing_pct = (cleaned_df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(8)

    rating_bins, balance_ratio, balance_verdict = rating_balance_summary(cleaned_df)

    row_lines = [
        f"- rows before cleaning: {stats.rows_before}",
        f"- rows after cleaning: {stats.rows_after}",
        f"- columns: {cleaned_df.shape[1]}",
    ]

    cleaning_lines = [
        f"- exact duplicate rows removed: {stats.dropped_exact_duplicates}",
        f"- rows removed due to missing core fields (`user_id`, `movie_id`, `rating`): {stats.dropped_missing_core_fields}",
    ]

    missing_lines = [f"- `{k}`: {v:.2f}%" for k, v in top_missing.items()]
    if not missing_lines:
        missing_lines = ["- no missing values detected"]

    balance_lines = []
    if rating_bins.empty:
        balance_lines.append("- could not compute rating balance")
    else:
        for cls_name, cls_count in rating_bins.items():
            balance_lines.append(f"- {cls_name}: {int(cls_count)}")
        balance_lines.append(f"- min/max class ratio: {balance_ratio:.3f}")
        balance_lines.append(f"- verdict: **{balance_verdict}**")

    figure_root = REPORTS_DIR
    fig_lines = []
    for path in figure_paths:
        relative_link = os.path.relpath(path, start=figure_root).replace("\\", "/")
        fig_lines.append(f"- ![]({relative_link})")

    if balance_verdict == "balanced":
        recommendation = (
            "Dataset has acceptable class spread for rating-based classification; "
            "no immediate balancing is required."
        )
    elif balance_verdict == "moderately-imbalanced":
        recommendation = (
            "Dataset is usable, but classification tasks should consider class weights "
            "or targeted resampling."
        )
    else:
        recommendation = (
            "Dataset is imbalanced for rating-class tasks; apply resampling or class-weighted "
            "training if you build a classifier."
        )

    report = f"""# Dataset Evaluation: {dataset_name}

## Source
- source file: `{source_path.relative_to(ROOT_DIR).as_posix()}`
- cleaned file: `{cleaned_path.relative_to(ROOT_DIR).as_posix()}`

## Size Snapshot
{chr(10).join(row_lines)}

## Cleaning Actions
{chr(10).join(cleaning_lines)}

## Missing Data (Top Fields)
{chr(10).join(missing_lines)}

## Rating Balance Check
{chr(10).join(balance_lines)}

## Visualizations
{chr(10).join(fig_lines)}

## Recommendation
{recommendation}
"""
    return report


def run_for_dataset(spec: DatasetSpec) -> tuple[Path, pd.DataFrame]:
    df = pd.read_csv(spec.source_path)
    cleaned_df, stats = clean_dataset(df)

    if spec.key == "movie_final_dataset":
        cleaned_df = adjust_half_step_ratings(cleaned_df)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(spec.cleaned_path, index=False)

    figure_paths = create_visualizations(cleaned_df, spec.key, FIGURES_DIR)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{spec.key}_evaluation.md"
    report = build_report(
        dataset_name=spec.key,
        source_path=spec.source_path,
        cleaned_path=spec.cleaned_path,
        cleaned_df=cleaned_df,
        stats=stats,
        figure_paths=figure_paths,
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path, cleaned_df


def main() -> None:
    datasets = [
        DatasetSpec(
            key="movie_final_dataset",
            source_path=DATA_DIR / "movie_final_dataset.csv",
            cleaned_path=CLEANED_DIR / "movie_final_dataset_cleaned.csv",
        ),
        DatasetSpec(
            key="trakt_ultimate_checkpoint",
            source_path=DATA_DIR / "trakt_ultimate_checkpoint.csv",
            cleaned_path=CLEANED_DIR / "trakt_ultimate_checkpoint_cleaned.csv",
        ),
    ]

    generated_reports: list[Path] = []
    cleaned_by_key: dict[str, pd.DataFrame] = {}
    for spec in datasets:
        report_path, cleaned_df = run_for_dataset(spec)
        generated_reports.append(report_path)
        cleaned_by_key[spec.key] = cleaned_df

    comparison_paths = create_comparison_visualizations(cleaned_by_key, COMPARISON_DIR)

    print("Generated reports:")
    for report_path in generated_reports:
        print(f"- {report_path.relative_to(ROOT_DIR).as_posix()}")
    print("Generated comparison figures:")
    for figure_path in comparison_paths:
        print(f"- {figure_path.relative_to(ROOT_DIR).as_posix()}")


if __name__ == "__main__":
    main()
