import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import sys
import io

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thiết lập đường dẫn
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = BASE_DIR / "outputs" / "data_split"
OUTPUT_DIR = BASE_DIR / "outputs" / "feature_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    start_time = time.time()
    print("🚀 Starting Advanced Feature Selection & Dimensionality Reduction...")

    # 1. Đọc dữ liệu từ data_split
    train_df = pd.read_csv(INPUT_DIR / "train.csv")
    val_df = pd.read_csv(INPUT_DIR / "val.csv")
    test_df = pd.read_csv(INPUT_DIR / "test.csv")

    # Xác định các cột tính năng (Features) và cột định danh (IDs)
    # Chúng ta sẽ thực hiện PCA trên các cột genre và các chỉ số rating
    genre_cols = [col for col in train_df.columns if col.startswith("genre_") and col not in ["genre_count_scaled"]]
    feature_cols = ["rating_scaled", "year_scaled", "genre_count_scaled"] + genre_cols
    id_cols = ["user_idx", "movie_idx", "user_id", "movie_id", "movie_title", "rating"]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    print(f"-> Original features: {X_train.shape[1]}")

    # 2. Xây dựng Pipeline xử lý (Theo slide yêu cầu)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),                 # Chuẩn hóa
        ('selector', VarianceThreshold(threshold=0.01)), # Loại bỏ các cột gần như không đổi
        ('pca', PCA(n_components=0.95, random_state=42)) # Giữ lại 95% phương sai thông tin
    ])

    # 3. Fit-on-train (CHỈ fit trên tập Train để tránh rò rỉ dữ liệu)
    pipeline.fit(X_train)

    # 4. Transform cho cả 3 tập
    X_train_reduced = pipeline.transform(X_train)
    X_val_reduced = pipeline.transform(X_val)
    X_test_reduced = pipeline.transform(X_test)

    # 5. Lưu Pipeline (.pkl) theo slide yêu cầu
    joblib.dump(pipeline, OUTPUT_DIR / "preprocess_pipeline.pkl")

    # 6. Tạo DataFrame kết quả (Gộp lại với IDs để sử dụng cho các bước sau)
    pca_cols = [f"pca_{i}" for i in range(X_train_reduced.shape[1])]
    
    train_reduced_df = pd.concat([train_df[id_cols].reset_index(drop=True), pd.DataFrame(X_train_reduced, columns=pca_cols)], axis=1)
    val_reduced_df = pd.concat([val_df[id_cols].reset_index(drop=True), pd.DataFrame(X_val_reduced, columns=pca_cols)], axis=1)
    test_reduced_df = pd.concat([test_df[id_cols].reset_index(drop=True), pd.DataFrame(X_test_reduced, columns=pca_cols)], axis=1)

    train_reduced_df.to_csv(OUTPUT_DIR / "train_features.csv", index=False)
    val_reduced_df.to_csv(OUTPUT_DIR / "val_features.csv", index=False)
    test_reduced_df.to_csv(OUTPUT_DIR / "test_features.csv", index=False)

    # 7. Lưu Artifacts (Theo slide yêu cầu)
    # Selected Features
    with open(OUTPUT_DIR / "selected_features.json", "w") as f:
        json.dump({"original_features": feature_cols, "num_pca_components": len(pca_cols)}, f, indent=4)

    # Explained Variance (PCA report)
    pca_model = pipeline.named_steps['pca']
    with open(OUTPUT_DIR / "explained_variance.json", "w") as f:
        json.dump({
            "total_variance_kept": float(np.sum(pca_model.explained_variance_ratio_)),
            "variance_per_component": pca_model.explained_variance_ratio_.tolist()
        }, f, indent=4)

    # Log
    end_time = time.time()
    log_data = {
        "fit_time_seconds": round(end_time - start_time, 4),
        "dimensions_before": X_train.shape[1],
        "dimensions_after": X_train_reduced.shape[1],
        "reduction_ratio": round(1 - (X_train_reduced.shape[1] / X_train.shape[1]), 4)
    }
    with open(OUTPUT_DIR / "log.json", "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"✅ DONE: Feature selection & Dimensionality Reduction completed!")
    print(f"-> Reduced features from {X_train.shape[1]} to {X_train_reduced.shape[1]}")
    print(f"-> Artifacts saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
