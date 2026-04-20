from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CleaningStats:
    rows_before: int
    rows_after: int
    dropped_exact_duplicates: int
    dropped_missing_core_fields: int


# ❌ ĐÃ BỎ timestamp
COMMON_SCHEMA_COLUMNS = [
    "user_id",
    "movie_id",
    "movie_title",
    "rating",
    "genres",
    "cast",
    "release_year",
    "language",
    "release_year_clean",
    "genres_list",
    "primary_genre",
    "genre_count",
]


# ================== GENRE CLEAN ==================
def _canonicalize_genre_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    alias_map = {
        "sci fi": "science fiction",
        "sci fi fantasy": "science fiction",
        "science fiction": "science fiction",
    }
    return alias_map.get(normalized, normalized)


def parse_genres(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [
            _canonicalize_genre_name(str(item["name"])) if isinstance(item, dict) else _canonicalize_genre_name(str(item))
            for item in value if item
        ]

    if isinstance(value, dict) and value.get("name"):
        return [_canonicalize_genre_name(str(value["name"]))]

    if pd.isna(value):
        return []

    raw = str(value).strip()
    if not raw:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, list):
                return [
                    _canonicalize_genre_name(str(item["name"])) if isinstance(item, dict) else _canonicalize_genre_name(str(item))
                    for item in parsed if item
                ]
        except Exception:
            pass

    parts = re.split(r"[,|;/]", raw)
    return [_canonicalize_genre_name(p.strip()) for p in parts if p.strip()]


# ================== NORMALIZE ==================
def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in ["rating", "vote_average", "vote_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "release_year" in df.columns:
        year_num = pd.to_numeric(df["release_year"], errors="coerce")
        year_num = year_num.where(year_num.between(1870, 2100), np.nan)

        year_parse = pd.to_datetime(df["release_year"], errors="coerce")
        df["release_year_clean"] = year_num.fillna(year_parse.dt.year).astype("Int64")

    if "language" in df.columns:
        df["language"] = df["language"].str.lower()

    if "genres" in df.columns:
        df["genres_list"] = df["genres"].apply(parse_genres)
        df["primary_genre"] = df["genres_list"].apply(lambda x: x[0] if x else None)
        df["genre_count"] = df["genres_list"].apply(len)

    return df


def harmonize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "cast_names" in df.columns:
        df = df.rename(columns={"cast_names": "cast"})

    for col in COMMON_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[COMMON_SCHEMA_COLUMNS].copy()


# ================== CLEAN MAIN ==================
def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    df = normalize_text_columns(df.copy())
    df = standardize_columns(df)
    df = harmonize_schema(df)

    rows_before = len(df)

    # ===== REMOVE EXACT DUPLICATES =====
    temp = df.copy()
    for col in temp.columns:
        if temp[col].apply(lambda x: isinstance(x, (list, dict, set))).any():
            temp[col] = temp[col].apply(lambda x: json.dumps(x, sort_keys=True))

    df = df.loc[~temp.duplicated()].copy()
    after_exact = len(df)

    # ===== REMOVE DUPLICATE USER-MOVIE (QUAN TRỌNG) =====
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    # ===== DROP MISSING =====
    required = ["user_id", "movie_id", "rating", "genres", "cast"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # ===== REMOVE GENRES RỖNG =====
    df = df[df["genres_list"].map(len) > 0]

    # ===== FILTER ENGLISH =====
    if "language" in df.columns:
        df = df[df["language"] == "en"]

    stats = CleaningStats(
        rows_before=rows_before,
        rows_after=len(df),
        dropped_exact_duplicates=rows_before - after_exact,
        dropped_missing_core_fields=after_exact - len(df),
    )

    return df, stats


# ================== RATING FIX ==================
def adjust_half_step_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rating = pd.to_numeric(df["rating"], errors="coerce")
    frac = rating - np.floor(rating)
    mask = np.isclose(frac, 0.5)
    df.loc[mask, "rating"] = rating.loc[mask] + 0.5
    return df


# ================== RUN ==================
if __name__ == "__main__":
    df = pd.read_csv(r"D:\Smart-Movie-Recommender\data\trakt_ultimate_checkpoint.csv")

    cleaned_df, stats = clean_dataset(df)
    cleaned_df = adjust_half_step_ratings(cleaned_df)

    print("\n=== FINAL SHAPE ===")
    print(cleaned_df.shape)

    print("\n=== STATS ===")
    print(stats)

    cleaned_df.to_csv("cleaned_final.csv", index=False)
    print("\nSaved: cleaned_final.csv")