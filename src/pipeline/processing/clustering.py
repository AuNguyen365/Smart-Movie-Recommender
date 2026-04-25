import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import json
import matplotlib.pyplot as plt
import time
import sys
import io

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thiết lập đường dẫn
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = BASE_DIR / "outputs" / "feature_selection" / "train_features.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "clustering"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    start_time = time.time()
    print("🚀 Starting Professional Movie Clustering...")

    # 1. Đọc dữ liệu đã qua PCA từ feature_selection
    df = pd.read_csv(INPUT_PATH)
    pca_cols = [col for col in df.columns if col.startswith("pca_")]
    
    # --- THAY ĐỔI QUAN TRỌNG: AGGREGATE THEO USER ---
    print("-> Aggregating user profiles (calculating mean interest vectors)...")
    user_profiles_raw = df.groupby('user_id')[pca_cols].mean().reset_index()
    X = user_profiles_raw[pca_cols]
    # -----------------------------------------------

    # 2. Định nghĩa số cụm (Dựa trên Silhouette tối ưu trước đó là 2)
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Fit model
    print(f"-> Fitting KMeans for USER CLUSTERING with k={k}...")
    kmeans.fit(X)
    labels = kmeans.labels_

    # 3. Tính toán Raw Metrics (Theo slide yêu cầu)
    print("-> Calculating metrics...")
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    metrics = {
        "silhouette_score": float(sil),
        "calinski_harabasz_score": float(ch),
        "davies_bouldin_score": float(db),
        "fit_time_seconds": round(time.time() - start_time, 4),
        "k_clusters": k
    }

    # 4. Tạo Cluster Profile 
    user_profiles_raw['cluster'] = labels
    
    # Tạo profile trung bình cho mỗi cụm người dùng
    cluster_summary = user_profiles_raw.groupby('cluster')[pca_cols].mean().reset_index()
    
    # Bổ sung thông tin kích thước cụm (số lượng người dùng trong mỗi nhóm)
    counts = user_profiles_raw['cluster'].value_counts().reset_index()
    counts.columns = ['cluster', 'user_count']
    cluster_summary = cluster_summary.merge(counts, on='cluster')

    # 5. Lưu Artifacts (Định dạng Parquet & PKL theo slide)
    # Lưu model
    joblib.dump(kmeans, MODEL_DIR / "clusterer.pkl")
    
    # Lưu Centroids
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=pca_cols)
    centroids_df['cluster'] = range(k)
    centroids_df.to_parquet(OUTPUT_DIR / "user_centroids.parquet", engine='fastparquet', index=False)
    
    # Lưu User Labels (Quan trọng nhất cho đề tài)
    user_labels_df = user_profiles_raw[['user_id', 'cluster']]
    user_labels_df.to_csv(OUTPUT_DIR / "user_clusters.csv", index=False)
    user_labels_df.to_parquet(OUTPUT_DIR / "user_clusters.parquet", engine='fastparquet', index=False)
    
    # Lưu Cluster Summary
    cluster_summary.to_parquet(OUTPUT_DIR / "user_cluster_profiles.parquet", engine='fastparquet', index=False)
    
    # Lưu Metrics
    with open(OUTPUT_DIR / "raw_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 6. Visualization (User Cluster Scatter Plot)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='coolwarm', alpha=0.7, edgecolors='w', s=60)
    
    # Vẽ centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=250, label='Group Centroids')
    
    plt.title(f'User Interests Clustering (k={k})', fontsize=14, fontweight='bold')
    plt.xlabel('Interest Dimension 0 (from PCA)')
    plt.ylabel('Interest Dimension 1 (from PCA)')
    plt.colorbar(scatter, label='User Group ID')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "user_cluster_viz.png")
    plt.close()

    print(f"✅ DONE: User Clustering completed!")
    print(f"-> Total users clustered: {len(user_profiles_raw)}")
    print(f"-> Metrics: Silhouette={sil:.4f}, CH={ch:.2f}, DB={db:.4f}")
    print(f"-> Artifacts saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()