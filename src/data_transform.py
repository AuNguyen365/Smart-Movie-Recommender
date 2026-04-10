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


COMMON_SCHEMA_COLUMNS = [
    "user_id",
    "movie_id",
    "movie_title",
    "rating",
    "genres",
    "cast",
    "release_year",
    "language",
    "timestamp",
    "release_year_clean",
    "genres_list",
    "primary_genre",
    "genre_count",
]


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

    # Fallback for plain text genres (e.g. Action|Thriller or Action, Thriller).
    text = raw_text.strip("[]")
    parts = [part.strip().strip("\"'") for part in re.split(r"[,|;/]", text)]
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


def harmonize_schema(df: pd.DataFrame) -> pd.DataFrame:
    harmonized = df.copy()

    # Align semantic-equivalent column names across sources.
    rename_map = {
        "cast_names": "cast",
    }
    source_renames = {k: v for k, v in rename_map.items() if k in harmonized.columns}
    if source_renames:
        harmonized = harmonized.rename(columns=source_renames)

    # Keep only the shared schema and create missing fields as NA.
    for col in COMMON_SCHEMA_COLUMNS:
        if col not in harmonized.columns:
            harmonized[col] = pd.NA

    harmonized = harmonized[COMMON_SCHEMA_COLUMNS].copy()
    return harmonized


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    cleaned = normalize_text_columns(df.copy())
    cleaned = standardize_columns(cleaned)
    cleaned = harmonize_schema(cleaned)

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

    if "language" in after_core.columns:
        after_core = after_core[
            after_core["language"].astype(str).str.lower().eq("en")
        ].copy()

    stats = CleaningStats(
        rows_before=rows_before,
        rows_after=len(after_core),
        dropped_exact_duplicates=dropped_exact_duplicates,
        dropped_missing_core_fields=dropped_missing_core_fields,
    )
    return after_core, stats


def adjust_half_step_ratings(df: pd.DataFrame) -> pd.DataFrame:
    adjusted = df.copy()
    if "rating" not in adjusted.columns:
        return adjusted

    rating = pd.to_numeric(adjusted["rating"], errors="coerce")
    fractional = rating - np.floor(rating)
    half_step_mask = np.isclose(fractional, 0.5)
    adjusted.loc[half_step_mask, "rating"] = rating.loc[half_step_mask] + 0.5
    return adjusted
