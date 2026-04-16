from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures" / "preprocessing"


def load_encoded_data() -> pd.DataFrame:
    path = CLEANED_DIR / "integrated_dataset_encoded.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Encoded dataset not found at {path}. Run preprocessing/encoding first."
        )
    return pd.read_csv(path)


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_rating_scaled_distribution(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "rating_scaled_distribution.png"
    plt.figure(figsize=(10, 5))
    if "rating_scaled" not in df.columns:
        plt.text(0.5, 0.5, "rating_scaled column not found", ha="center", va="center")
        plt.axis("off")
    else:
        df["rating_scaled"].dropna().plot(kind="hist", bins=20, color="#3B82F6", edgecolor="white")
        plt.title("Rating Scaled Distribution")
        plt.xlabel("rating_scaled")
        plt.ylabel("Count")
    _save_plot(path)
    return path


def plot_genre_count_distribution(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "genre_count_distribution.png"
    plt.figure(figsize=(10, 5))
    if "genre_count" not in df.columns:
        plt.text(0.5, 0.5, "genre_count column not found", ha="center", va="center")
        plt.axis("off")
    else:
        df["genre_count"].dropna().astype(float).plot(kind="hist", bins=15, color="#10B981", edgecolor="white")
        plt.title("Genre Count Distribution")
        plt.xlabel("genre_count")
        plt.ylabel("Count")
    _save_plot(path)
    return path


def plot_source_distribution(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "source_distribution.png"
    plt.figure(figsize=(10, 5))
    if "source_encoded" not in df.columns:
        plt.text(0.5, 0.5, "source_encoded column not found", ha="center", va="center")
        plt.axis("off")
    else:
        source_counts = df["source_encoded"].fillna(-1).astype(int).value_counts().sort_index()
        labels = {0: "movie_final_dataset", 1: "trakt_ultimate_checkpoint", -1: "missing"}
        source_counts.index = [labels.get(int(idx), str(idx)) for idx in source_counts.index]
        source_counts.plot(kind="bar", color="#F59E0B")
        plt.title("Source Distribution")
        plt.xlabel("Source")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
    _save_plot(path)
    return path


def plot_primary_genre_distribution(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "primary_genre_distribution.png"
    plt.figure(figsize=(12, 5))
    if "primary_genre" not in df.columns:
        plt.text(0.5, 0.5, "primary_genre column not found", ha="center", va="center")
        plt.axis("off")
    else:
        top_genres = df["primary_genre"].fillna("unknown").value_counts().head(12)
        top_genres.sort_values().plot(kind="barh", color="#8B5CF6")
        plt.title("Top Primary Genres")
        plt.xlabel("Count")
        plt.ylabel("Primary Genre")
    _save_plot(path)
    return path


def plot_missing_value_distribution(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "missing_value_distribution.png"
    plt.figure(figsize=(12, 5))

    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
    else:
        missing_pct.plot(kind="bar", color="#6B7280")
        plt.title("Missing Value Ratio by Column")
        plt.xlabel("Column")
        plt.ylabel("Missing (%)")
        plt.xticks(rotation=45, ha="right")

    _save_plot(path)
    return path


def plot_top_encoded_genres(df: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "top_encoded_genres.png"
    plt.figure(figsize=(12, 5))
    genre_cols = [
        col
        for col in df.columns
        if col.startswith("genre_") and col not in {"genre_count", "genre_count_scaled"}
    ]
    if not genre_cols:
        plt.text(0.5, 0.5, "encoded genre columns not found", ha="center", va="center")
        plt.axis("off")
    else:
        genre_totals = df[genre_cols].sum().sort_values(ascending=False).head(15)
        genre_totals.index = [col.replace("genre_", "") for col in genre_totals.index]
        genre_totals.sort_values().plot(kind="barh", color="#EF4444")
        plt.title("Top Encoded Genres")
        plt.xlabel("Count")
        plt.ylabel("Genre")
    _save_plot(path)
    return path


def main() -> None:
    df = load_encoded_data()
    figure_paths = [
        plot_missing_value_distribution(df),
        plot_rating_scaled_distribution(df),
        plot_genre_count_distribution(df),
        plot_source_distribution(df),
        plot_primary_genre_distribution(df),
        plot_top_encoded_genres(df),
    ]

    print("Generated preprocessing visualizations:")
    for path in figure_paths:
        print(f"- {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
