# BÁO CÁO PHÂN CỤM PHIM (CLUSTERING)

## 1. Tổng quan

- **Số lượng dữ liệu**: 8,328 phim
- **Số features**: 29
- **Số cụm tối ưu**: 8
- **Thuật toán**: K-Means Clustering

## 2. Đánh giá chất lượng

- **Silhouette Score**: 0.1901
- **Davies-Bouldin Index**: 1.7215
- **Calinski-Harabasz Score**: 650.78

## 3. Phân bố các cụm

- **Cụm 0**: 1,778 phim (21.3%)
- **Cụm 1**: 1,703 phim (20.4%)
- **Cụm 2**: 681 phim (8.2%)
- **Cụm 3**: 1,933 phim (23.2%)
- **Cụm 4**: 81 phim (1.0%)
- **Cụm 5**: 266 phim (3.2%)
- **Cụm 6**: 1,465 phim (17.6%)
- **Cụm 7**: 421 phim (5.1%)

## 4. Giải thích PCA

- **PC1**: Giải thích 15.15% phương sai
- **PC2**: Giải thích 7.70% phương sai
- **Tổng**: 22.84%

## 5. Files đã tạo

### Models
- `kmeans_model.pkl` - Model K-Means đã train
- `clustering_scaler.pkl` - StandardScaler để chuẩn hóa
- `clustering_pca.pkl` - PCA để giảm chiều

### Dữ liệu
- `cluster_train_labeled.csv` - Dữ liệu gốc + cluster labels
- `cluster_profiles.csv` - Đặc điểm từng cụm

### Biểu đồ
- `optimal_k_analysis.png` - Phân tích chọn K tối ưu
- `clusters_2d_visualization.png` - Visualize các cụm 2D
- `cluster_distribution.png` - Phân bố số lượng phim
- `correlation_matrix.png` - Ma trận tương quan features

## 6. Ứng dụng cho hệ thống gợi ý

Các cụm phim này có thể được sử dụng để:

1. **Gợi ý trong cùng cụm**: Nếu người dùng thích phim A thuộc cụm X, 
   gợi ý các phim khác trong cụm X.

2. **Phân khúc người dùng**: Xác định người dùng thuộc nhóm sở thích nào 
   dựa trên lịch sử xem phim.

3. **Cold-start problem**: Với phim mới, phân loại vào cụm phù hợp để 
   gợi ý cho đúng nhóm người dùng.

4. **Đa dạng hóa gợi ý**: Gợi ý phim từ nhiều cụm khác nhau để tăng 
   tính đa dạng.

## 7. Kết luận

Đã hoàn thành phân cụm 8,328 phim thành 8 nhóm với chất lượng TRUNG BÌNH. 
Kết quả có thể tích hợp vào hệ thống gợi ý để cải thiện độ chính xác và đa dạng.
