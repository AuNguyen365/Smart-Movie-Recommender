import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Đường dẫn dữ liệu đã encode
DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "cleaned" / "integrated_dataset_encoded.csv"

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH)
# Loại bỏ duplicate column nếu có
df = df.loc[:, ~df.columns.duplicated()]

genre_cols = [
    col for col in df.columns
    if col.startswith("genre_") and col not in ["genre_count_scaled"]
]
# Chia train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Recommendation features
rec_df = train_df[["user_idx", "movie_idx", "rating_scaled"]].copy()

# Clustering features
cluster_features = [
    "rating_scaled", "year_scaled", "genre_count_scaled"
] + genre_cols
cluster_df = train_df[["user_idx"] + cluster_features].copy()
if "genre_count" in cluster_df.columns:
    cluster_df = cluster_df.drop(columns=["genre_count"])

# Association features
assoc_df = train_df[genre_cols].copy()

# (Optional) Lưu các tập đặc thù ra outputs nếu cần
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
rec_df.to_csv(OUTPUT_DIR / "rec_train.csv", index=False)
cluster_df.to_csv(OUTPUT_DIR / "cluster_train.csv", index=False)
assoc_df.to_csv(OUTPUT_DIR / "assoc_train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)