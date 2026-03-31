from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
CLEANED_DIR = DATA_DIR / "cleaned"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source_path: Path
    cleaned_path: Path


@dataclass
class CleaningStats:
    rows_before: int
    rows_after: int
    dropped_exact_duplicates: int
    dropped_missing_core_fields: int


def parse_genres(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]).strip().lower())
            elif isinstance(item, str):
                names.append(item.strip().lower())
        return [x for x in names if x]

    if isinstance(value, dict):
        if value.get("name"):
            return [str(value["name"]).strip().lower()]
        return []

    if pd.isna(value):
        return []

    raw_text = value.strip() if isinstance(value, str) else str(value).strip()
    if not raw_text:
        return []

    # First try JSON format: [{"id": ..., "name": "Crime"}, ...]
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw_text)
            if isinstance(parsed, list):
                names: list[str] = []
                for item in parsed:
                    if isinstance(item, dict) and item.get("name"):
                        names.append(str(item["name"]).strip().lower())
                    elif isinstance(item, str):
                        names.append(item.strip().lower())
                names = [x for x in names if x]
                if names:
                    return names
        except Exception:
            pass

    # Fallback for comma-separated format.
    text = raw_text.strip("[]")
    parts = [part.strip().strip("\"'") for part in text.split(",")]
    return [part.lower() for part in parts if part]


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [c.strip() for c in normalized.columns]

    numeric_candidates = ["rating", "vote_average", "vote_count"]
    for col in numeric_candidates:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if "release_year" in normalized.columns:
        release_as_num = pd.to_numeric(normalized["release_year"], errors="coerce")
        release_as_num = release_as_num.where(
            release_as_num.between(1870, 2100), np.nan
        )
        release_parsed = pd.to_datetime(normalized["release_year"], errors="coerce")
        normalized["release_year_clean"] = release_as_num.fillna(release_parsed.dt.year)
        normalized["release_year_clean"] = normalized["release_year_clean"].astype("Int64")

    if "timestamp" in normalized.columns:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")

    if "language" in normalized.columns:
        normalized["language"] = normalized["language"].str.lower()

    if "genres" in normalized.columns:
        normalized["genres_list"] = normalized["genres"].apply(parse_genres)
        normalized["primary_genre"] = normalized["genres_list"].apply(
            lambda x: x[0] if x else "unknown"
        )
        normalized["genre_count"] = normalized["genres_list"].apply(len)

    return normalized


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    cleaned = normalize_text_columns(df.copy())
    cleaned = standardize_columns(cleaned)

    rows_before = len(cleaned)
    dedupe_view = cleaned.copy()
    for col in dedupe_view.columns:
        has_nested = dedupe_view[col].map(
            lambda x: isinstance(x, (list, dict, set))
        ).any()
        if has_nested:
            dedupe_view[col] = dedupe_view[col].map(
                lambda x: json.dumps(sorted(x), sort_keys=True)
                if isinstance(x, set)
                else json.dumps(x, sort_keys=True)
                if isinstance(x, (list, dict, set))
                else x
            )

    dedupe_mask = dedupe_view.duplicated()
    deduped = cleaned.loc[~dedupe_mask].copy()
    dropped_exact_duplicates = rows_before - len(deduped)

    core_cols = [c for c in ["user_id", "movie_id", "rating"] if c in deduped.columns]
    if core_cols:
        after_core = deduped.dropna(subset=core_cols).copy()
    else:
        after_core = deduped.copy()
    dropped_missing_core_fields = len(deduped) - len(after_core)

    stats = CleaningStats(
        rows_before=rows_before,
        rows_after=len(after_core),
        dropped_exact_duplicates=dropped_exact_duplicates,
        dropped_missing_core_fields=dropped_missing_core_fields,
    )
    return after_core, stats


def _plot_missing_values(df: pd.DataFrame, out_dir: Path) -> Path:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(15)

    plt.figure(figsize=(10, 5))
    if missing_pct.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
    else:
        missing_pct.plot(kind="bar")
        plt.ylabel("Missing (%)")
        plt.title("Top Missing Fields")
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = out_dir / "missing_values.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_rating_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    plt.figure(figsize=(10, 5))
    if "rating" not in df.columns:
        plt.text(0.5, 0.5, "rating column not found", ha="center", va="center")
        plt.axis("off")
    else:
        rating_counts = df["rating"].dropna().value_counts().sort_index()
        rating_counts.plot(kind="bar")
        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
    plt.tight_layout()

    path = out_dir / "rating_distribution.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_top_genres(df: pd.DataFrame, out_dir: Path) -> Path:
    plt.figure(figsize=(10, 5))
    if "genres_list" not in df.columns:
        plt.text(0.5, 0.5, "genres column not found", ha="center", va="center")
        plt.axis("off")
    else:
        exploded = df["genres_list"].explode().dropna()
        top_genres = exploded.value_counts().head(10)
        if top_genres.empty:
            plt.text(0.5, 0.5, "No genres parsed", ha="center", va="center")
            plt.axis("off")
        else:
            top_genres.sort_values().plot(kind="barh")
            plt.title("Top 10 Genres")
            plt.xlabel("Count")
    plt.tight_layout()

    path = out_dir / "top_genres.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_language_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    if "language" not in df.columns:
        plt.text(0.5, 0.5, "language column not found", ha="center", va="center")
        plt.axis("off")
    else:
        language_counts = df["language"].fillna("unknown").value_counts().head(10)
        language_counts.plot(kind="bar")
        plt.title("Top Languages")
        plt.xlabel("Language")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
    plt.tight_layout()

    path = out_dir / "language_distribution.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_release_year_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    plt.figure(figsize=(10, 5))
    if "release_year_clean" not in df.columns:
        plt.text(0.5, 0.5, "release_year_clean not found", ha="center", va="center")
        plt.axis("off")
    else:
        year_counts = (
            df["release_year_clean"].dropna().astype(int).value_counts().sort_index().tail(20)
        )
        if year_counts.empty:
            plt.text(0.5, 0.5, "No release year available", ha="center", va="center")
            plt.axis("off")
        else:
            year_counts.plot(kind="bar")
            plt.title("Release Year Distribution (last 20 years found)")
            plt.xlabel("Year")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = out_dir / "release_year_distribution.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def create_visualizations(df: pd.DataFrame, dataset_key: str) -> list[Path]:
    out_dir = FIGURES_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        _plot_missing_values(df, out_dir),
        _plot_rating_distribution(df, out_dir),
        _plot_top_genres(df, out_dir),
        _plot_language_distribution(df, out_dir),
        _plot_release_year_distribution(df, out_dir),
    ]
    return outputs


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


def run_for_dataset(spec: DatasetSpec) -> Path:
    df = pd.read_csv(spec.source_path)
    cleaned_df, stats = clean_dataset(df)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(spec.cleaned_path, index=False)

    figure_paths = create_visualizations(cleaned_df, spec.key)

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
    return report_path


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
    for spec in datasets:
        generated_reports.append(run_for_dataset(spec))

    print("Generated reports:")
    for report_path in generated_reports:
        print(f"- {report_path.relative_to(ROOT_DIR).as_posix()}")


if __name__ == "__main__":
    main()
