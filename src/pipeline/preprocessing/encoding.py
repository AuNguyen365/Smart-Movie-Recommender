from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = ROOT_DIR / "outputs"
ENCODERS_DIR = OUTPUT_DIR / "encoders"


def load_data() -> pd.DataFrame:
    input_path = CLEANED_DIR / "integrated_dataset_cleaned.csv"
    print(f"Loading data from {input_path}...")
    return pd.read_csv(input_path)


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, sp.csr_matrix]:
    print("Step 1: Parse genres and cast...")
    df["genres_list_parsed"] = df["genres_list"].apply(
        lambda x: json.loads(x) if pd.notna(x) else []
    )
    df["cast"] = df["cast"].fillna("unknown")

    print("Step 2: Min-Max Scaling for rating...")
    scaler_rating = MinMaxScaler()
    df["rating_scaled"] = scaler_rating.fit_transform(df[["rating"]])
    joblib.dump(scaler_rating, ENCODERS_DIR / "minmax_scaler_rating.pkl")

    print("Step 3: Multi-Label Binarizer for genres_list...")
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df["genres_list_parsed"])
    genre_cols = [f"genre_{g.replace(' ', '_')}" for g in mlb.classes_]
    genre_df = pd.DataFrame(genre_matrix, columns=genre_cols, index=df.index)
    df = pd.concat([df, genre_df], axis=1)
    joblib.dump(mlb, ENCODERS_DIR / "multilabel_binarizer_genres.pkl")

    print("Step 4: Label Encoding for primary_genre...")
    le_genre = LabelEncoder()
    df["primary_genre_encoded"] = le_genre.fit_transform(
        df["primary_genre"].fillna("unknown")
    )
    joblib.dump(le_genre, ENCODERS_DIR / "label_encoder_primary_genre.pkl")

    print("Step 5: Label Encoding for user_id and movie_id...")
    le_user = LabelEncoder()
    le_movie = LabelEncoder()
    df["user_idx"] = le_user.fit_transform(df["user_id"].astype(str))
    df["movie_idx"] = le_movie.fit_transform(df["movie_id"].astype(str))
    joblib.dump(le_user, ENCODERS_DIR / "label_encoder_user.pkl")
    joblib.dump(le_movie, ENCODERS_DIR / "label_encoder_movie.pkl")

    print("Step 6: TF-IDF Encoding for cast...")
    tfidf_cast = TfidfVectorizer(max_features=500, token_pattern=r"[^,]+")
    cast_matrix = tfidf_cast.fit_transform(df["cast"])
    joblib.dump(tfidf_cast, ENCODERS_DIR / "tfidf_cast_vectorizer.pkl")

    print("Step 7: Binary Encoding for source...")
    df["source_encoded"] = df["source"].map(
        {"movie_final_dataset": 0, "trakt_ultimate_checkpoint": 1}
    )

    print("Step 8: Min-Max Scaling for release_year and genre_count...")
    scaler_meta = MinMaxScaler()
    df[["year_scaled", "genre_count_scaled"]] = scaler_meta.fit_transform(
        df[["release_year", "genre_count"]].fillna(0)
    )
    joblib.dump(scaler_meta, ENCODERS_DIR / "minmax_scaler_meta.pkl")

    # Drop temporary parsing column
    df = df.drop(columns=["genres_list_parsed"])

    return df, cast_matrix


def main() -> None:
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    encoded_df, cast_matrix = preprocess_features(df)

    output_csv = CLEANED_DIR / "integrated_dataset_encoded.csv"
    output_npz = ENCODERS_DIR / "cast_tfidf_matrix.npz"

    print(f"Step 9: Saving results...")
    encoded_df.to_csv(output_csv, index=False)
    sp.save_npz(output_npz, cast_matrix)

    print(f"Done! Saved main dataset to {output_csv.relative_to(ROOT_DIR)} with shape {encoded_df.shape}")
    print(f"Saved cast TF-IDF matrix to {output_npz.relative_to(ROOT_DIR)} with shape {cast_matrix.shape}")
    print(f"Saved encoders to {ENCODERS_DIR.relative_to(ROOT_DIR)}/")


if __name__ == "__main__":
    main()
