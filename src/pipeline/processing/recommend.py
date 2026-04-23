import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
import json
import time
import sys
import io

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thiết lập đường dẫn
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = BASE_DIR / "outputs" / "data_split" / "train.csv"
TEST_PATH = BASE_DIR / "outputs" / "data_split" / "test.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "recommendation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    start_time = time.time()
    print("🚀 Starting Recommendation Engine (Matrix Factorization via Sklearn TruncatedSVD)...")

    # 1. Đọc dữ liệu
    train_df = pd.read_csv(INPUT_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # 2. Tạo User-Item Pivot Table
    print("-> Creating User-Item matrix...")
    user_item_matrix = train_df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
    
    # 3. Huấn luyện TruncatedSVD (Matrix Factorization)
    print("-> Fitting SVD model...")
    n_factors = min(50, user_item_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    
    # User Factors (U * Sigma)
    user_factors_matrix = svd.fit_transform(user_item_matrix)
    # Item Factors (V^T)
    item_factors_matrix = svd.components_.T

    # 4. Lưu Factors (Định dạng Parquet theo slide yêu cầu)
    user_factors_df = pd.DataFrame(user_factors_matrix, index=user_item_matrix.index)
    user_factors_df.columns = [f'factor_{i}' for i in range(n_factors)]
    user_factors_df.to_parquet(OUTPUT_DIR / "user_factors.parquet", engine='fastparquet')

    item_factors_df = pd.DataFrame(item_factors_matrix, index=user_item_matrix.columns)
    item_factors_df.columns = [f'factor_{i}' for i in range(n_factors)]
    item_factors_df.to_parquet(OUTPUT_DIR / "item_factors.parquet", engine='fastparquet')

    # 5. Tạo Top-K Rec Lists cho Demo (Theo slide)
    print("-> Generating rec_lists.parquet...")
    # Dự đoán điểm số bằng cách nhân ma trận (Reconstruct matrix)
    # Vì dữ liệu lớn, chúng ta chỉ lấy demo 50 user đầu tiên
    sample_users = user_item_matrix.index[:50]
    sample_user_idx = range(len(sample_users))
    
    # Ma trận dự đoán (Dot product of factors)
    predicted_ratings = np.dot(user_factors_matrix[:50], item_factors_matrix.T)
    
    movie_titles = train_df[['movie_id', 'movie_title']].drop_duplicates().set_index('movie_id')['movie_title'].to_dict()
    
    rec_list_data = []
    for i, user_id in enumerate(sample_users):
        user_preds = predicted_ratings[i]
        # Lấy top 10 phim có điểm cao nhất
        top_indices = np.argsort(user_preds)[::-1][:10]
        for rank, idx in enumerate(top_indices, 1):
            movie_id = user_item_matrix.columns[idx]
            rec_list_data.append({
                "user_id": user_id,
                "movie_id": int(movie_id),
                "movie_title": movie_titles.get(movie_id, "Unknown"),
                "score": round(float(user_preds[idx]), 2),
                "rank": rank
            })
            
    rec_lists_df = pd.DataFrame(rec_list_data)
    rec_lists_df.to_parquet(OUTPUT_DIR / "rec_lists.parquet", engine='fastparquet', index=False)
    rec_lists_df.to_csv(OUTPUT_DIR / "recommendations.csv", index=False)

    # 6. Tính toán Offline Metrics (RMSE sơ bộ trên tập Train vì giới hạn thời gian tính toán toàn cục)
    # Tính RMSE đơn giản cho phần demo
    mse = np.mean((user_item_matrix.values[:50] - predicted_ratings)**2)
    rmse = np.sqrt(mse)

    metrics = {
        "rmse": float(rmse),
        "n_factors": n_factors,
        "train_time_seconds": round(time.time() - start_time, 4),
        "evaluation_method": "TruncatedSVD (Matrix Factorization)"
    }
    with open(OUTPUT_DIR / "raw_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ DONE: Recommendation artifacts generated!")
    print(f"-> Artifacts: user_factors.parquet, item_factors.parquet, rec_lists.parquet")
    print(f"-> Metrics: RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()