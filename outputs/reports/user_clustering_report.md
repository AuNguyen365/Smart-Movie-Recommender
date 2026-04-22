# BÁO CÁO PHÂN CỤM NGƯỜI DÙNG (USER CLUSTERING)

## 1. Tổng quan

- **Số lượng người dùng**: 2,958 users
- **Số features**: 28
- **Số cụm tối ưu**: 2
- **Thuật toán**: K-Means Clustering

## 2. Đánh giá chất lượng

- **Silhouette Score**: 0.1921
- **Davies-Bouldin Index**: 2.6623
- **Calinski-Harabasz Score**: 243.07

## 3. Phân bố các cụm

- **Cụm 0**: 620 users (21.0%)
- **Cụm 1**: 2,338 users (79.0%)

## 4. Giải thích PCA

- **PC1**: Giải thích 10.37% phương sai
- **PC2**: Giải thích 7.87% phương sai
- **Tổng**: 18.24%

## 5. Files đã tạo

### Models
- `kmeans_users.pkl` - Model K-Means đã train
- `user_clustering_scaler.pkl` - StandardScaler để chuẩn hóa
- `user_clustering_pca.pkl` - PCA để giảm chiều

### Dữ liệu
- `users_clustered.csv` - Users đã gán cụm
- `user_cluster_profiles.csv` - Đặc điểm từng cụm

### Biểu đồ
- `optimal_k_analysis.png` - Phân tích chọn K tối ưu
- `clusters_2d_visualization.png` - Visualize các cụm 2D
- `cluster_distribution.png` - Phân bố số lượng users
- `correlation_matrix.png` - Ma trận tương quan features

## 6. Ứng dụng cho hệ thống gợi ý

Các cụm người dùng này có thể được sử dụng để:

1. **Gợi ý theo nhóm sở thích**: Nếu user A thuộc cụm X, 
   gợi ý phim mà users khác trong cụm X thích.

2. **Phân khúc marketing**: Xác định nhóm khách hàng mục tiêu 
   cho các chiến dịch quảng cáo phim.

3. **Cold-start problem**: Với user mới, phân loại vào cụm phù hợp 
   dựa trên vài đánh giá đầu tiên.

4. **Personalization**: Tùy chỉnh giao diện và nội dung theo 
   đặc điểm từng cụm người dùng.

## 7. Kết luận

Đã hoàn thành phân cụm 2,958 người dùng thành 2 nhóm với 
chất lượng TRUNG BÌNH. 
Kết quả có thể tích hợp vào hệ thống gợi ý để cải thiện độ chính xác và cá nhân hóa.
