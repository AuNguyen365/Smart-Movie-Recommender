import pandas as pd
import numpy as np
import json
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
DATA_PATH = BASE_DIR / "outputs" / "data_split" / "train.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "clustering"
MODELS_DIR = OUTPUT_DIR / "models"

# Tạo thư mục nếu chưa có
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class MovieClusteringPipeline:
    """Pipeline hoàn chỉnh cho phân cụm người dùng"""
    
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
        
        # Lưu metrics
        self.metrics = {}
        
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
        
        # Lưu thông tin cơ bản vào metrics
        self.metrics['num_users'] = len(self.df_user_profiles)
        self.metrics['num_features'] = self.df.shape[1]
        
        return self
    
    def scale_data(self):
        """Bước 2: Chuẩn hóa dữ liệu"""
        self.X_scaled = self.scaler.fit_transform(self.df.values)
        return self
    
    def find_optimal_k(self, k_range=range(2, 11)):
        """Bước 3: Tìm số cụm tối ưu"""
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.X_scaled, labels))
            calinski_scores.append(calinski_harabasz_score(self.X_scaled, labels))
        
        self.best_k = k_range[np.argmax(silhouette_scores)]
        
        # Lưu tất cả metrics cho các giá trị K
        self.metrics['k_analysis'] = {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'calinski_harabasz_scores': calinski_scores,
            'best_k': self.best_k,
            'best_silhouette': max(silhouette_scores)
        }
        
        return self
    
    def perform_clustering(self):
        """Bước 4: Thực hiện phân cụm với K tối ưu"""
        self.kmeans_model = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        self.labels = self.kmeans_model.fit_predict(self.X_scaled)
        
        # Lưu phân bố các cụm
        unique, counts = np.unique(self.labels, return_counts=True)
        self.metrics['cluster_distribution'] = {
            f'cluster_{int(cid)}': int(count) 
            for cid, count in zip(unique, counts)
        }
        self.metrics['cluster_percentages'] = {
            f'cluster_{int(cid)}': float(count / len(self.labels) * 100)
            for cid, count in zip(unique, counts)
        }
        
        return self
    
    def evaluate_clustering(self):
        """Bước 5: Đánh giá kết quả phân cụm"""
        self.metrics['silhouette_score'] = float(silhouette_score(self.X_scaled, self.labels))
        self.metrics['davies_bouldin_score'] = float(davies_bouldin_score(self.X_scaled, self.labels))
        self.metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(self.X_scaled, self.labels))
        
        # Đánh giá chất lượng
        if self.metrics['silhouette_score'] > 0.5:
            self.metrics['quality'] = 'Tốt'
        elif self.metrics['silhouette_score'] > 0.3:
            self.metrics['quality'] = 'Trung bình'
        else:
            self.metrics['quality'] = 'Yếu'
        
        return self
    
    def apply_pca(self, n_components=2):
        """Bước 6: Giảm chiều dữ liệu bằng PCA"""
        self.pca = PCA(n_components=n_components, random_state=42)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.explained_var = self.pca.explained_variance_ratio_
        
        # Lưu thông tin PCA
        self.metrics['pca'] = {
            'n_components': n_components,
            'explained_variance_ratio': self.explained_var.tolist(),
            'total_explained_variance': float(sum(self.explained_var))
        }
        
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
                'percentage': len(cluster_data) / len(self.df_user_profiles) * 100,
                'avg_rating': cluster_data['avg_rating'].mean(),
                'avg_num_ratings': cluster_data['num_ratings'].mean(),
                'avg_rating_std': cluster_data['rating_std'].mean()
            }
            
            # Thêm top 5 thể loại (lưu đầy đủ để phân tích)
            genre_pref_cols = [col for col in cluster_data.columns if col.endswith('_pref')]
            if genre_pref_cols:
                genre_means = cluster_data[genre_pref_cols].mean().sort_values(ascending=False)
                top_5_genres = genre_means.head(5)
                
                profile['top_genres'] = []
                for genre, score in top_5_genres.items():
                    genre_name = genre.replace('genre_', '').replace('_pref', '')
                    profile['top_genres'].append({
                        'genre': genre_name,
                        'preference_score': float(score)
                    })
            
            cluster_profiles.append(profile)
        
        # Lưu cluster profiles dưới dạng Parquet
        profiles_df = pd.DataFrame(cluster_profiles)
        
        # Xử lý cột top_genres (list of dicts) để lưu vào Parquet
        if 'top_genres' in profiles_df.columns:
            # Chuyển list of dicts thành JSON string để lưu Parquet
            profiles_df['top_genres'] = profiles_df['top_genres'].apply(json.dumps)
        
        profiles_df.to_parquet(OUTPUT_DIR / 'user_cluster_profiles.parquet', index=False)
        
        # Lưu dữ liệu users với labels dưới dạng Parquet
        df_with_clusters.to_parquet(OUTPUT_DIR / 'users_clustered.parquet', index=False)
        
        # Lưu thông tin cluster profiles vào metrics
        self.metrics['cluster_profiles'] = cluster_profiles
        
        return self
    
    def save_models(self):
        """Bước 8: Lưu models"""
        joblib.dump(self.kmeans_model, MODELS_DIR / 'kmeans_users.pkl')
        joblib.dump(self.scaler, MODELS_DIR / 'user_clustering_scaler.pkl')
        joblib.dump(self.pca, MODELS_DIR / 'user_clustering_pca.pkl')
        return self
    
    def save_metrics(self):
        """Bước 9: Lưu raw metrics dưới dạng JSON"""
        metrics_path = OUTPUT_DIR / 'clustering_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
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
             .save_models()
             .save_metrics())
        
        return self


def main():
    """Hàm main để chạy pipeline"""
    pipeline = MovieClusteringPipeline(DATA_PATH)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()