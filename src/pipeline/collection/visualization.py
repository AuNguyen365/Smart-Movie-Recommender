from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def create_visualizations(
    df: pd.DataFrame, dataset_key: str, figures_dir: Path
) -> list[Path]:
    out_dir = figures_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        _plot_missing_values(df, out_dir),
        _plot_rating_distribution(df, out_dir),
        _plot_top_genres(df, out_dir),
        _plot_language_distribution(df, out_dir),
        _plot_release_year_distribution(df, out_dir),
    ]
    return outputs


def _plot_comparison_rating_distribution(
    datasets: dict[str, pd.DataFrame], out_dir: Path
) -> Path:
    rating_indexes: set[float] = set()
    counts_by_dataset: dict[str, pd.Series] = {}

    for name, df in datasets.items():
        if "rating" not in df.columns:
            counts_by_dataset[name] = pd.Series(dtype=int)
            continue
        counts = df["rating"].dropna().value_counts().sort_index()
        counts_by_dataset[name] = counts
        rating_indexes.update(counts.index.tolist())

    plt.figure(figsize=(12, 6))
    if not rating_indexes:
        plt.text(0.5, 0.5, "No rating data available", ha="center", va="center")
        plt.axis("off")
    else:
        x = np.array(sorted(rating_indexes), dtype=float)
        labels = [str(int(v)) if float(v).is_integer() else str(v) for v in x]
        width = 0.35
        offsets = np.linspace(
            -(len(datasets) - 1) * width / 2,
            (len(datasets) - 1) * width / 2,
            len(datasets),
        )

        for (offset, (name, counts)) in zip(offsets, counts_by_dataset.items()):
            y = [int(counts.get(v, 0)) for v in x]
            plt.bar(np.arange(len(x)) + offset, y, width=width, label=name)

        plt.xticks(np.arange(len(x)), labels)
        plt.title("Rating Distribution Comparison")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.legend()
    plt.tight_layout()

    path = out_dir / "rating_distribution_comparison.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_comparison_language_distribution(
    datasets: dict[str, pd.DataFrame], out_dir: Path
) -> Path:
    lang_counts_by_dataset: dict[str, pd.Series] = {}
    all_languages: pd.Series = pd.Series(dtype=int)

    for name, df in datasets.items():
        if "language" not in df.columns:
            lang_counts_by_dataset[name] = pd.Series(dtype=int)
            continue
        counts = df["language"].fillna("unknown").value_counts()
        lang_counts_by_dataset[name] = counts
        all_languages = all_languages.add(counts, fill_value=0)

    top_languages = all_languages.sort_values(ascending=False).head(10).index.tolist()

    plt.figure(figsize=(12, 6))
    if not top_languages:
        plt.text(0.5, 0.5, "No language data available", ha="center", va="center")
        plt.axis("off")
    else:
        x = np.arange(len(top_languages))
        width = 0.35
        offsets = np.linspace(
            -(len(datasets) - 1) * width / 2,
            (len(datasets) - 1) * width / 2,
            len(datasets),
        )

        for (offset, (name, counts)) in zip(offsets, lang_counts_by_dataset.items()):
            y = [int(counts.get(lang, 0)) for lang in top_languages]
            plt.bar(x + offset, y, width=width, label=name)

        plt.xticks(x, top_languages, rotation=0)
        plt.title("Top Languages Comparison")
        plt.xlabel("Language")
        plt.ylabel("Count")
        plt.legend()
    plt.tight_layout()

    path = out_dir / "language_distribution_comparison.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_comparison_release_year_distribution(
    datasets: dict[str, pd.DataFrame], out_dir: Path
) -> Path:
    year_counts_by_dataset: dict[str, pd.Series] = {}
    year_index: set[int] = set()

    for name, df in datasets.items():
        if "release_year_clean" not in df.columns:
            year_counts_by_dataset[name] = pd.Series(dtype=int)
            continue
        counts = (
            df["release_year_clean"].dropna().astype(int).value_counts().sort_index()
        )
        year_counts_by_dataset[name] = counts
        year_index.update(counts.index.tolist())

    plt.figure(figsize=(12, 6))
    if not year_index:
        plt.text(0.5, 0.5, "No release year data available", ha="center", va="center")
        plt.axis("off")
    else:
        years = sorted(year_index)
        # Keep plot readable by focusing on the latest 20 years present.
        years = years[-20:]
        for name, counts in year_counts_by_dataset.items():
            y = [int(counts.get(year, 0)) for year in years]
            plt.plot(years, y, marker="o", label=name)

        plt.title("Release Year Distribution Comparison (latest 20 years)")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.xticks(years, rotation=45, ha="right")
        plt.legend()
    plt.tight_layout()

    path = out_dir / "release_year_distribution_comparison.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def _plot_comparison_missing_values(
    datasets: dict[str, pd.DataFrame], out_dir: Path
) -> Path:
    missing_pct_by_dataset: dict[str, pd.Series] = {}
    missing_union = pd.Series(dtype=float)

    for name, df in datasets.items():
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        missing_pct_by_dataset[name] = missing_pct
        missing_union = missing_union.add(missing_pct, fill_value=0)

    top_fields = missing_union.sort_values(ascending=False)
    top_fields = top_fields[top_fields > 0].head(10).index.tolist()

    plt.figure(figsize=(12, 6))
    if not top_fields:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
    else:
        x = np.arange(len(top_fields))
        width = 0.35
        offsets = np.linspace(
            -(len(datasets) - 1) * width / 2,
            (len(datasets) - 1) * width / 2,
            len(datasets),
        )

        for (offset, (name, missing_pct)) in zip(offsets, missing_pct_by_dataset.items()):
            y = [float(missing_pct.get(field, 0.0)) for field in top_fields]
            plt.bar(x + offset, y, width=width, label=name)

        plt.xticks(x, top_fields, rotation=45, ha="right")
        plt.title("Top Missing Fields Comparison")
        plt.ylabel("Missing (%)")
        plt.legend()
    plt.tight_layout()

    path = out_dir / "missing_values_comparison.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def create_comparison_visualizations(
    datasets: dict[str, pd.DataFrame], comparison_dir: Path
) -> list[Path]:
    comparison_dir.mkdir(parents=True, exist_ok=True)

    return [
        _plot_comparison_missing_values(datasets, comparison_dir),
        _plot_comparison_rating_distribution(datasets, comparison_dir),
        _plot_comparison_language_distribution(datasets, comparison_dir),
        _plot_comparison_release_year_distribution(datasets, comparison_dir),
    ]
