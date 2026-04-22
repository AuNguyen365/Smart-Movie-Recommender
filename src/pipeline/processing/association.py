import sys
import io
import pandas as pd
import json
import ast  # Thư viện để đọc chuỗi list (VD: "['action', 'comedy']")
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Fix tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("1. Đang tải và biến đổi dữ liệu...")
file_path = r'C:\Users\Windows\Smart-Movie-Recommender\data\cleaned\integrated_dataset_encoded.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại: {file_path}")
    sys.exit()

# CẢI THIỆN 1: Nâng tiêu chuẩn yêu thích để giảm nhiễu (noise)
df_liked = df[df['rating'] >= 8.0]

basket = df_liked.groupby(['user_id', 'movie_title'])['rating'].count().unstack().reset_index().fillna(0).set_index('user_id')

def encode_units(x):
    return True if x >= 1 else False

if hasattr(basket, 'map'):
    basket_sets = basket.map(encode_units)
else:
    basket_sets = basket.applymap(encode_units)

print(f"-> Đã tạo ma trận với kích thước: {basket_sets.shape[0]} Users và {basket_sets.shape[1]} Phim.")

# ==========================================
# 1.5 CHUẨN BỊ TỪ ĐIỂN THỂ LOẠI ĐỂ ÉP CẶP
# ==========================================
print("\n[Tính năng mới] Đang xây dựng bộ lọc Thể loại (Domain Constraint)...")
movie_genres = {}
# Lấy cột movie_title và genres_list
for _, row in df.drop_duplicates(subset=['movie_title']).iterrows():
    try:
        # Biến chuỗi "['action', 'drama']" thành một Set thực sự trong Python
        movie_genres[row['movie_title']] = set(ast.literal_eval(row['genres_list']))
    except:
        movie_genres[row['movie_title']] = set()

# ==========================================
# 2. CHẠY THUẬT TOÁN FP-GROWTH
# ==========================================
print("\n2. Đang tìm tập phổ biến bằng FP-Growth...")
frequent_itemsets = fpgrowth(basket_sets, min_support=0.003, use_colnames=True)

if len(frequent_itemsets) == 0:
    print("❌ Dữ liệu quá thưa, hãy giảm min_support.")
    sys.exit()

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(f"-> Tìm thấy {len(frequent_itemsets)} tập phim phổ biến.")

# ==========================================
# 3. SINH VÀ LỌC LUẬT (PRUNE & DOMAIN CONSTRAINT)
# ==========================================
print("\n3. Đang tạo và lọc luật kết hợp (Association Rules)...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# CẢI THIỆN 2: Lọc theo Leverage và Conviction (Loại bỏ luật ngẫu nhiên)
# Vô hạn (inf) conviction thường xảy ra khi confidence = 1.0, ta giới hạn lại để tránh lỗi toán học
import numpy as np
rules.replace([np.inf, -np.inf], 1000, inplace=True) 

filtered_rules = rules[
    (rules['lift'] > 1.5) & 
    (rules['confidence'] > 0.1) &
    (rules['conviction'] > 1.2)
]

print(f"-> Số luật trước khi ép Thể loại: {len(filtered_rules)}")

# CẢI THIỆN 3: ÉP CẶP THEO THỂ LOẠI (Ràng buộc Domain)
def check_genre_similarity(row):
    ants = list(row['antecedents'])
    cons = list(row['consequents'])
    
    # Lấy tất cả thể loại của nhóm phim gốc (A)
    ant_genres = set.union(*[movie_genres.get(a, set()) for a in ants]) if ants else set()
    # Lấy tất cả thể loại của nhóm phim gợi ý (B)
    con_genres = set.union(*[movie_genres.get(c, set()) for c in cons]) if cons else set()
    
    # Nếu hai nhóm phim có ÍT NHẤT 1 thể loại chung -> Trả về True (Giữ luật)
    return len(ant_genres.intersection(con_genres)) > 0

# Áp dụng bộ lọc
filtered_rules['is_valid_domain'] = filtered_rules.apply(check_genre_similarity, axis=1)
filtered_rules = filtered_rules[filtered_rules['is_valid_domain'] == True]

# Bỏ cột cờ đi cho sạch data
filtered_rules = filtered_rules.drop(columns=['is_valid_domain'])

print(f"-> Số luật Siêu Chính Xác (sau khi ép Thể loại): {len(filtered_rules)}")

# ==========================================
# 4. LƯU ARTIFACTS / OUTPUT
# ==========================================
print("\n4. Đang xuất các file kết quả (Artifacts)...")
output_dir = r'C:\Users\Windows\Smart-Movie-Recommender\data\cleaned'

if len(filtered_rules) > 0:
    frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: list(x))
    itemsets_path = f"{output_dir}\\itemsets.parquet"
    frequent_itemsets[['itemsets_str', 'support', 'length']].to_parquet(itemsets_path)

    filtered_rules['antecedents_str'] = filtered_rules['antecedents'].apply(lambda x: list(x))
    filtered_rules['consequents_str'] = filtered_rules['consequents'].apply(lambda x: list(x))

    cols_to_save = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'leverage', 'conviction']
    rules_path = f"{output_dir}\\rules.parquet"
    filtered_rules[cols_to_save].to_parquet(rules_path)

    movie_dict = pd.Series(df['movie_title'].values, index=df['movie_id'].astype(str)).to_dict()
    dict_path = f"{output_dir}\\dictionary.json"
    with open(dict_path, 'w', encoding='utf-8') as f:
        json.dump(movie_dict, f, ensure_ascii=False, indent=4)

    print("\n[RAW METRICS] Top 5 luật gợi ý CHÍNH XÁC NHẤT (Theo Lift):")
    print(filtered_rules.sort_values('lift', ascending=False)[['antecedents_str', 'consequents_str', 'lift']].head())
else:
    print("❌ Không còn luật nào sau khi ép Thể loại. Hãy hạ min_support hoặc rating xuống một chút!")

print("\n🎉 --- HOÀN TẤT THÀNH CÔNG! --- 🎉")