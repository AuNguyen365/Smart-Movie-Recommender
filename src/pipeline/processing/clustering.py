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
        self.df_user_profiles = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.pca = None
        self.best_k = None
        self.kmeans_model = None
        self.labels = None
        
    def load_data(self):
        """Bước 1: Load dữ liệu và tạo user profiles"""
        df_interactions = pd.read_csv(self.data_path)
        
        # Tạo user profiles: Tổng hợp đặc điểm của mỗi user
        user_profiles = []
        
        for user_id in df_interactions['user_idx'].unique():
            user_data = df_interactions[df_interactions['user_idx'] == user_id]
            
            profile = {
                'user_idx': user_id,
                'avg_rating': user_data['rating_scaled'].mean(),
                'num_ratings': len(user_data),
                'rating_std': user_data['rating_scaled'].std(),
            }
            
            # Tính tỷ lệ xem từng thể loại
            genre_cols = [col for col in user_data.columns if col.startswith('genre_')]
            for genre in genre_cols:
                profile[f'{genre}_pref'] = user_data[genre].mean()
            
            user_profiles.append(profile)
        
        self.df_user_profiles = pd.DataFrame(user_profiles)
        
        # Loại bỏ user_idx và chuẩn bị dữ liệu cho clustering
        self.df = self.df_user_profiles.drop('user_idx', axis=1).fillna(0)
        
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
        """Bước 7: Phân tích đặc điểm từng cụm người dùng"""
        # Thêm labels vào user profiles
        df_with_clusters = self.df_user_profiles.copy()
        df_with_clusters['user_cluster'] = self.labels
        
        # Phân tích từng cụm
        cluster_profiles = []
        
        for cluster_id in range(self.best_k):
            cluster_data = df_with_clusters[df_with_clusters['user_cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'num_users': len(cluster_data),
                'percentage': len(cluster_data)/len(self.df_user_profiles)*100,
                'avg_rating': cluster_data['avg_rating'].mean(),
                'avg_num_ratings': cluster_data['num_ratings'].mean(),
                'avg_rating_std': cluster_data['rating_std'].mean()
            }
            
            # Thêm top 3 thể loại
            genre_pref_cols = [col for col in cluster_data.columns if col.endswith('_pref')]
            if genre_pref_cols:
                genre_means = cluster_data[genre_pref_cols].mean().sort_values(ascending=False)
                top_3_genres = genre_means.head(3)
                for i, (genre, score) in enumerate(top_3_genres.items(), 1):
                    genre_name = genre.replace('genre_', '').replace('_pref', '')
                    profile[f'top_genre_{i}'] = genre_name
                    profile[f'top_genre_{i}_score'] = score
            
            cluster_profiles.append(profile)
        
        # Lưu cluster profiles
        profiles_df = pd.DataFrame(cluster_profiles)
        profiles_df.to_csv(OUTPUT_DIR / 'user_cluster_profiles.csv', index=False)
        
        # Lưu dữ liệu users với labels
        df_with_clusters.to_csv(OUTPUT_DIR / 'users_clustered.csv', index=False)
        
        return self
    
    def save_models(self):
        """Bước 8: Lưu models"""
        joblib.dump(self.kmeans_model, MODELS_DIR / 'kmeans_users.pkl')
        joblib.dump(self.scaler, MODELS_DIR / 'user_clustering_scaler.pkl')
        joblib.dump(self.pca, MODELS_DIR / 'user_clustering_pca.pkl')
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
        
        return self


def main():
    """Hàm main để chạy pipeline"""
    pipeline = MovieClusteringPipeline(DATA_PATH)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()