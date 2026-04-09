# Kế hoạch Encoding – Smart Movie Recommender

## Tổng quan dữ liệu đầu vào

| Thông tin | Giá trị |
|---|---|
| File | `data/cleaned/integrated_dataset_cleaned.csv` |
| Số dòng | **10,659 rows** |
| Số cột | **14 cột** (13 + source) |
| Số phim riêng biệt | 2,720 movies |
| Số người dùng riêng biệt | 3,535 users |
| Số thể loại riêng biệt | **27 genres** |
| Rating range | 1.0 → 10.0 (10 giá trị) |

---

## Phân tích từng cột cần Encoding

| Cột | Kiểu hiện tại | Vấn đề | Giải pháp Encoding |
|---|---|---|---|
| `rating` | float64 (1.0–10.0) | Thang điểm cần chuẩn hóa | **Min-Max Scaling** → [0, 1] |
| `genres_list` | object (JSON string) | Nhiều thể loại/phim, không phải số | **Multi-Label Binarizer** → 27 cột 0/1 |
| `primary_genre` | object (string) | ~19 giá trị categorical | **Label Encoding** → số nguyên |
| `genre_count` | int64 | Khoảng giá trị cần chuẩn hóa | **Min-Max Scaling** → [0, 1] |
| `user_id` | object (string tên người dùng) | Không phải số | **Label Encoding** → số nguyên |
| `movie_id` | int64 | ID không liên tục, khoảng cách lớn | **Label Encoding** (remap 0→N) |
| `cast` | object (CSV string diễn viên) | Text, không phải số | **TF-IDF Vectorizer** → vector số |
| `release_year_clean` | int64 | Khoảng giá trị lớn | **Min-Max Scaling** → [0, 1] |
| `source` | object (2 giá trị) | Categorical nhị phân | **Binary Encoding** → 0/1 |
| `language` | object | Đã lọc chỉ còn `en` → không cần | ❌ Bỏ qua |
| `movie_title`, `genres`, `timestamp` | object | Dùng để tra cứu/hiển thị | ❌ Giữ nguyên, không encode |

---

## Các bước thực hiện chi tiết

### Bước 1 – Chuẩn bị (Pre-encoding)

**Mục tiêu:** Load và chuẩn bị dữ liệu thô trước khi encode.

```python
import pandas as pd
import json

df = pd.read_csv('data/cleaned/integrated_dataset_cleaned.csv')

# 1.1 Parse genres_list từ JSON string → Python list
df['genres_list_parsed'] = df['genres_list'].apply(
    lambda x: json.loads(x) if pd.notna(x) else []
)

# 1.2 Chuẩn bị cột cast (đã là string dạng "Actor A, Actor B, ...")
df['cast'] = df['cast'].fillna('unknown')
```

---

### Bước 2 – Min-Max Scaling cho `rating`

**Kỹ thuật:** `MinMaxScaler` (sklearn)

```
rating gốc: 1.0 → 10.0
rating mới: 0.0 → 1.0

Công thức: (x - min) / (max - min)
Ví dụ: rating = 7 → (7 - 1) / (10 - 1) = 0.667
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler_rating = MinMaxScaler()
df['rating_scaled'] = scaler_rating.fit_transform(df[['rating']])
```

**Lý do chọn Min-Max thay vì Z-score:**
- Rating có khoảng cố định (1–10), không có outlier
- Collaborative Filtering cần giá trị trong [0, 1] để tính cosine similarity

---

### Bước 3 – Multi-Label Binarizer cho `genres_list`

**Kỹ thuật:** `MultiLabelBinarizer` (sklearn)

```
Một phim có thể thuộc nhiều thể loại → tạo 27 cột nhị phân

Ví dụ:
  genres_list = ['action', 'thriller', 'crime']
  → genre_action=1, genre_thriller=1, genre_crime=1, genre_drama=0, ...
```

```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['genres_list_parsed'])
genre_cols   = [f'genre_{g.replace(" ", "_")}' for g in mlb.classes_]

genre_df = pd.DataFrame(genre_matrix, columns=genre_cols, index=df.index)
df = pd.concat([df, genre_df], axis=1)

print(f"Tạo {len(genre_cols)} cột genre: {genre_cols}")
```

**27 thể loại sẽ tạo ra 27 cột:**
```
genre_action, genre_adventure, genre_animation, genre_biography,
genre_comedy, genre_crime, genre_documentary, genre_drama,
genre_family, genre_fantasy, genre_history, genre_horror,
genre_music, genre_mystery, genre_romance, genre_science_fiction,
genre_thriller, genre_war, ...
```

---

### Bước 4 – Label Encoding cho `primary_genre`

**Kỹ thuật:** `LabelEncoder` (sklearn)

```
Ánh xạ thể loại chính thành số nguyên:
  action    → 0
  adventure → 1
  animation → 2
  comedy    → 3
  drama     → 4
  horror    → 5
  thriller  → 6
  ...
```

```python
from sklearn.preprocessing import LabelEncoder

le_genre = LabelEncoder()
df['primary_genre_encoded'] = le_genre.fit_transform(
    df['primary_genre'].fillna('unknown')
)

# Xem ánh xạ
mapping = dict(zip(le_genre.classes_, le_genre.transform(le_genre.classes_)))
print("Genre mapping:", mapping)
```

> **Lưu ý:** Label Encoding tạo ra thứ tự ngầm (drama=3 > comedy=2). Chỉ dùng trong Decision Tree, Random Forest. Tránh dùng trực tiếp trong Linear Models.

---

### Bước 5 – Label Encoding cho `user_id` và `movie_id`

**Kỹ thuật:** `LabelEncoder` – Ánh xạ sang số nguyên liên tục

```
user_id gốc: 'Sejian', 'MovieGuys', 'Dean', ...  (3,535 users)
            → user_idx: 0, 1, 2, ... 3534

movie_id gốc: 12345, 67890, ... (số lớn, không liên tục)
            → movie_idx: 0, 1, 2, ... 2719
```

```python
le_user  = LabelEncoder()
le_movie = LabelEncoder()

df['user_idx']  = le_user.fit_transform(df['user_id'])
df['movie_idx'] = le_movie.fit_transform(df['movie_id'].astype(str))

print(f"Num users: {df['user_idx'].nunique()}")   # 3535
print(f"Num movies: {df['movie_idx'].nunique()}")  # 2720
```

**Lý do quan trọng:**
- Collaborative Filtering (SVD, NMF, ALS) cần index liên tục từ 0
- User-Item Matrix shape sẽ là (3535, 2720)

---

### Bước 6 – TF-IDF Encoding cho `cast`

**Kỹ thuật:** `TfidfVectorizer` (sklearn)

```
Mỗi bộ phim → chuỗi diễn viên → vector số (500 chiều)

TF-IDF ưu tiên:
  - Diễn viên xuất hiện ÍT nhưng đặc trưng → trọng số CAO
  - Diễn viên xuất hiện NHIỀU (phổ biến) → trọng số THẤP

Ví dụ:
  cast = "Tom Hanks, Robin Wright, Gary Sinise"
  → vector 500 entries với các trọng số TF-IDF
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# Dùng dấu phẩy làm separator thay vì khoảng trắng
tfidf_cast = TfidfVectorizer(
    max_features=500,      # Chỉ lấy 500 diễn viên quan trọng nhất
    token_pattern=r'[^,]+' # Tách bằng dấu phẩy
)
cast_matrix = tfidf_cast.fit_transform(df['cast'])
# cast_matrix: sparse matrix (10659 × 500)

# Lưu sparse matrix
sp.save_npz('outputs/encoders/cast_tfidf_matrix.npz', cast_matrix)
print(f"Cast TF-IDF matrix shape: {cast_matrix.shape}")
```

**Không concat vào df** (sparse matrix 10659×500 quá lớn cho CSV)

---

### Bước 7 – Binary Encoding cho `source`

**Kỹ thuật:** Map thủ công (2 giá trị → 0/1)

```
movie_final_dataset          → 0
trakt_ultimate_checkpoint    → 1
```

```python
df['source_encoded'] = df['source'].map({
    'movie_final_dataset': 0,
    'trakt_ultimate_checkpoint': 1
})
```

---

### Bước 8 – Min-Max Scaling cho `release_year_clean` và `genre_count`

```python
scaler_meta = MinMaxScaler()
df[['year_scaled', 'genre_count_scaled']] = scaler_meta.fit_transform(
    df[['release_year_clean', 'genre_count']].fillna(0)
)
```

---

### Bước 9 – Lưu tất cả kết quả

**9.1 DataFrame chính → CSV**

```python
df.to_csv('data/cleaned/integrated_dataset_encoded.csv', index=False)
```

**9.2 Lưu các encoder/scaler → .pkl** (dùng lại khi predict)

```python
import joblib
from pathlib import Path

Path('outputs/encoders').mkdir(parents=True, exist_ok=True)

joblib.dump(le_user,       'outputs/encoders/label_encoder_user.pkl')
joblib.dump(le_movie,      'outputs/encoders/label_encoder_movie.pkl')
joblib.dump(le_genre,      'outputs/encoders/label_encoder_primary_genre.pkl')
joblib.dump(mlb,           'outputs/encoders/multilabel_binarizer_genres.pkl')
joblib.dump(tfidf_cast,    'outputs/encoders/tfidf_cast_vectorizer.pkl')
joblib.dump(scaler_rating, 'outputs/encoders/minmax_scaler_rating.pkl')
joblib.dump(scaler_meta,   'outputs/encoders/minmax_scaler_meta.pkl')

print("Đã lưu tất cả encoders.")
```

---

## Tóm tắt output cuối cùng

### File `integrated_dataset_encoded.csv` – các cột mới thêm vào

| Cột mới | Kiểu | Mô tả |
|---|---|---|
| `rating_scaled` | float [0,1] | Rating đã chuẩn hóa |
| `user_idx` | int 0–3534 | Index người dùng |
| `movie_idx` | int 0–2719 | Index phim |
| `primary_genre_encoded` | int | Thể loại chính dạng số |
| `source_encoded` | int 0/1 | Nguồn dữ liệu |
| `year_scaled` | float [0,1] | Năm phát hành chuẩn hóa |
| `genre_count_scaled` | float [0,1] | Số thể loại chuẩn hóa |
| `genre_action` | int 0/1 | Multi-label: action |
| `genre_drama` | int 0/1 | Multi-label: drama |
| `genre_thriller` | int 0/1 | Multi-label: thriller |
| ... 27 cột `genre_*` | int 0/1 | Toàn bộ 27 thể loại |

**Tổng số cột sau encoding:** 14 (cũ) + 7 (scaled/encoded) + 27 (genre) = **~48 cột**

### File `cast_tfidf_matrix.npz` – sparse matrix (10,659 × 500)

### Thư mục `outputs/encoders/` – 7 file .pkl

---

## Thứ tự ưu tiên thực hiện

```
🔴 CAO – Bắt buộc cho Collaborative Filtering:
   ✅ Bước 5 → user_idx, movie_idx
   ✅ Bước 2 → rating_scaled

🟠 CAO – Bắt buộc cho Content-Based Filtering:
   ✅ Bước 3 → genre_* columns (27 cột)
   ✅ Bước 6 → TF-IDF cast matrix

🟡 TRUNG BÌNH – Bổ sung features cho model:
   ✅ Bước 4 → primary_genre_encoded
   ✅ Bước 8 → year_scaled, genre_count_scaled

🟢 THẤP – Metadata / phân tích:
   ✅ Bước 7 → source_encoded
```

---

## Sơ đồ luồng

```
integrated_dataset_cleaned.csv (10,659 × 14)
          │
          ├─ rating ──────────────── MinMaxScaler ──────────────→ rating_scaled
          │
          ├─ genres_list ─────────── MultiLabelBinarizer ────────→ genre_* (×27)
          │
          ├─ primary_genre ────────── LabelEncoder ───────────────→ primary_genre_encoded
          │
          ├─ user_id ──────────────── LabelEncoder ───────────────→ user_idx
          │
          ├─ movie_id ─────────────── LabelEncoder ───────────────→ movie_idx
          │
          ├─ cast ─────────────────── TF-IDF Vectorizer ──────────→ cast_tfidf_matrix.npz
          │
          ├─ release_year_clean ───── MinMaxScaler ──────────────→ year_scaled
          │
          ├─ genre_count ──────────── MinMaxScaler ──────────────→ genre_count_scaled
          │
          └─ source ───────────────── Binary Map ─────────────────→ source_encoded
                    │
                    ▼
    integrated_dataset_encoded.csv (~48 cột)
    outputs/encoders/ (7 file .pkl)
```

---

## File thực hiện

```
src/preprocessing.py      ← Viết toàn bộ logic tại đây
outputs/encoders/         ← Lưu encoder .pkl
data/cleaned/
  └── integrated_dataset_encoded.csv
```
