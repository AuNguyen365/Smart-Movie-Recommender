import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Thiết lập đường dẫn
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "outputs" / "cluster_train.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"

# Tạo thư mục nếu chưa có
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class MovieClusteringPipeline:    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.pca = None
        self.best_k = None
        self.kmeans_model = None
        self.labels = None
        
    def load_data(self):
        """Bước 1: Load dữ liệu"""
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        return self
    
    def scale_data(self):
        """Bước 2: Chuẩn hóa dữ liệu"""
        self.X_scaled = self.scaler.fit_transform(self.df.values)
        return self
    
    def find_optimal_k(self, k_range=range(2, 11)):
        """Bước 3: Tìm số cụm tối ưu"""
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            silhouette_scores.append(silhouette_score(self.X_scaled, labels))
        
        self.best_k = k_range[np.argmax(silhouette_scores)]
        return self
    
    def perform_clustering(self):
        """Bước 4: Thực hiện phân cụm với K tối ưu"""
        self.kmeans_model = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        self.labels = self.kmeans_model.fit_predict(self.X_scaled)
        return self
    
    def evaluate_clustering(self):
        """Bước 5: Đánh giá kết quả phân cụm"""
        self.silhouette = silhouette_score(self.X_scaled, self.labels)
        self.davies_bouldin = davies_bouldin_score(self.X_scaled, self.labels)
        self.calinski = calinski_harabasz_score(self.X_scaled, self.labels)
        return self
    
    def apply_pca(self, n_components=2):
        """Bước 6: Giảm chiều dữ liệu bằng PCA"""
        self.pca = PCA(n_components=n_components, random_state=42)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.explained_var = self.pca.explained_variance_ratio_
        return self
    
    def analyze_clusters(self):
        """Bước 7: Phân tích đặc điểm từng cụm"""
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = self.labels
        
        # Phân tích từng cụm
        self.cluster_info = []
        for cluster_id in range(self.best_k):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            info = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data)/len(self.df)*100
            }
            
            if 'rating_scaled' in cluster_data.columns:
                info['avg_rating'] = cluster_data['rating_scaled'].mean()
            if 'year_scaled' in cluster_data.columns:
                info['avg_year'] = cluster_data['year_scaled'].mean()
            
            # Top thể loại
            genre_cols = [col for col in cluster_data.columns if col.startswith('genre_')]
            if genre_cols:
                genre_means = cluster_data[genre_cols].mean().sort_values(ascending=False)
                info['top_genres'] = [g.replace('genre_', '').replace('_', ' ').title() 
                                      for g in genre_means.head(3).index]
            
            self.cluster_info.append(info)
        
        # Lưu dữ liệu
        df_with_clusters.to_csv(OUTPUT_DIR / 'cluster_train_labeled.csv', index=False)
        
        return self
    
    def save_models(self):
        """Bước 8: Lưu models"""
        joblib.dump(self.kmeans_model, MODELS_DIR / 'kmeans_model.pkl')
        joblib.dump(self.scaler, MODELS_DIR / 'clustering_scaler.pkl')
        joblib.dump(self.pca, MODELS_DIR / 'clustering_pca.pkl')
        return self
    
    def run_full_pipeline(self):
        """Chạy toàn bộ pipeline"""
        (self.load_data()
             .scale_data()
             .find_optimal_k()
             .perform_clustering()
             .evaluate_clustering()
             .apply_pca()
             .analyze_clusters()
             .save_models())
        
        # Hiển thị kết quả
        self.print_summary()
        
        return self
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("KẾT QUẢ PHÂN CỤM PHIM")
        print("=" * 70)
        
        print(f"\nTỔNG QUAN:")
        print(f"   • Số lượng phim: {len(self.df):,}")
        print(f"   • Số features: {self.df.shape[1]}")
        print(f"   • Số cụm tối ưu: {self.best_k}")
        
        print(f"\nCHẤT LƯỢNG PHÂN CỤM:")
        print(f"   • Silhouette Score: {self.silhouette:.4f} {'(Tốt)' if self.silhouette > 0.5 else '(Trung bình)' if self.silhouette > 0.3 else '(Yếu)'}")
        print(f"   • Davies-Bouldin Index: {self.davies_bouldin:.4f}")
        print(f"   • Calinski-Harabasz Score: {self.calinski:.2f}")
        
        print(f"\nGIẢM CHIỀU (PCA):")
        print(f"   • PC1: {self.explained_var[0]*100:.2f}% variance")
        print(f"   • PC2: {self.explained_var[1]*100:.2f}% variance")
        print(f"   • Tổng: {sum(self.explained_var)*100:.2f}%")
        
        print(f"\nPHÂN BỐ CÁC CỤM:")
        for info in self.cluster_info:
            print(f"   • Cụm {info['cluster_id']}: {info['size']:,} phim ({info['percentage']:.1f}%)", end="")
            if 'avg_rating' in info:
                print(f" | Rating: {info['avg_rating']:.3f}", end="")
            if 'top_genres' in info:
                print(f" | Top thể loại: {', '.join(info['top_genres'])}", end="")
            print()
        
        print(f"\nFILE ĐÃ LƯU:")
        print(f"   • Models: {MODELS_DIR}/kmeans_model.pkl, clustering_scaler.pkl, clustering_pca.pkl")
        print(f"   • Dữ liệu: {OUTPUT_DIR}/cluster_train_labeled.csv")


def main():
    pipeline = MovieClusteringPipeline(DATA_PATH)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()