from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"


def _resolve_cleaned_input(*filenames: str) -> Path:
    for name in filenames:
        candidate = CLEANED_DIR / name
        if candidate.exists():
            return candidate
    return CLEANED_DIR / filenames[0]


def integrate_datasets(
    movie_path: Path,
    trakt_path: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    d1 = pd.read_csv(movie_path)
    d2 = pd.read_csv(trakt_path)

    d1["source"] = "TMDB_cleaned"
    d2["source"] = "Trakt_cleaned"

    merged = pd.concat([d1, d2], ignore_index=True)
    before = len(merged)

    merged = merged.drop_duplicates()
    after_exact = len(merged)

    key_cols = [
        c for c in ["user_id", "movie_id", "rating", "timestamp"] if c in merged.columns
    ]
    if key_cols:
        merged = merged.drop_duplicates(subset=key_cols, keep="first")
    after_key = len(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    stats: dict[str, int | str] = {
        "tmdb_rows": len(d1),
        "trakt_rows": len(d2),
        "merged_before_dedup": before,
        "merged_after_exact_dedup": after_exact,
        "merged_after_key_dedup": after_key,
        # Keep the original script output text unchanged.
        "output": output_path.relative_to(ROOT_DIR).as_posix(),
    }
    return merged, stats


def main() -> None:
    p1 = _resolve_cleaned_input("TMDB_cleaned.csv", "movie_final_dataset_cleaned.csv")
    p2 = _resolve_cleaned_input(
        "Trakt_cleaned.csv",
        "Trakt_cleaned .csv",
        "trakt_ultimate_checkpoint_cleaned.csv",
    )
    out = CLEANED_DIR / "integrated_dataset_cleaned.csv"

    _, stats = integrate_datasets(p1, p2, out)

    # Match original print order and style from scripts/integrate_datasets.py.
    print("tmdb_rows", stats["tmdb_rows"])
    print("trakt_rows", stats["trakt_rows"])
    print("merged_before_dedup", stats["merged_before_dedup"])
    print("merged_after_exact_dedup", stats["merged_after_exact_dedup"])
    print("merged_after_key_dedup", stats["merged_after_key_dedup"])
    print("output", stats["output"])


if __name__ == "__main__":
    main()
