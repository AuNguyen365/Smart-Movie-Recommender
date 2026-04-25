import pandas as pd
from pathlib import Path
from mlxtend.frequent_patterns import fpgrowth, association_rules
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
OUTPUT_DIR = BASE_DIR / "outputs" / "association"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    start_time = time.time()
    print("🚀 Starting Professional Association Rule Mining (FP-Growth)...")

    # 1. Đọc dữ liệu
    df = pd.read_csv(INPUT_PATH)
    
    # Chỉ lấy các phim được đánh giá cao (>= 7.0 hoặc tương đương 3.5/5)
    high_rated = df[df['rating'] >= 7.0].copy()
    
    # 2. Tạo Transaction Matrix (Cấu trúc nhị phân cho FP-Growth)
    print("-> Creating transaction matrix...")
    basket = (high_rated.groupby(['user_id', 'movie_title'])['rating']
              .count().unstack().reset_index().fillna(0)
              .set_index('user_id'))
    
    # Convert sang boolean (True/False)
    basket_sets = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)

    # 3. Chạy FP-Growth (Tìm Itemsets)
    print("-> Finding frequent itemsets...")
    frequent_itemsets = fpgrowth(basket_sets, min_support=0.003, use_colnames=True)
    
    # 4. Khai phá luật 
    print("-> Mining association rules...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Tính toán thêm metrics nếu cần (mlxtend đã có sẵn support, confidence, lift, leverage, conviction)
    
    # 5. Lưu Artifacts 
    # Xử lý antecedents/consequents sang string để lưu parquet được
    rules_save = rules.copy()
    rules_save['antecedents'] = rules_save['antecedents'].apply(lambda x: list(x))
    rules_save['consequents'] = rules_save['consequents'].apply(lambda x: list(x))
    
    # Chuyển list sang string vì Parquet không hỗ trợ list lồng nhau một cách đơn giản trong một số engine
    rules_save['antecedents_str'] = rules_save['antecedents'].astype(str)
    rules_save['consequents_str'] = rules_save['consequents'].astype(str)
    
    # Lưu Rules Parquet
    rules_save.to_parquet(OUTPUT_DIR / "rules.parquet", engine='fastparquet', index=False)
    
    # Lưu Itemsets Parquet
    frequent_itemsets_save = frequent_itemsets.copy()
    frequent_itemsets_save['itemsets'] = frequent_itemsets_save['itemsets'].apply(lambda x: list(x)).astype(str)
    frequent_itemsets_save.to_parquet(OUTPUT_DIR / "itemsets.parquet", engine='fastparquet', index=False)

    # Lưu Dictionary (Mapping ID -> Title)
    movie_dict = df[['movie_id', 'movie_title']].drop_duplicates().set_index('movie_id')['movie_title'].to_dict()
    with open(OUTPUT_DIR / "dictionary.json", "w", encoding='utf-8') as f:
        json.dump(movie_dict, f, ensure_ascii=False, indent=4)

    # Lưu Raw Metrics
    metrics = {
        "mining_time_seconds": round(time.time() - start_time, 4),
        "num_rules": len(rules),
        "num_itemsets": len(frequent_itemsets),
        "avg_lift": float(rules['lift'].mean()) if len(rules) > 0 else 0
    }
    with open(OUTPUT_DIR / "raw_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Lưu CSV dự phòng cho API
    rules_save.to_csv(OUTPUT_DIR / "rules.csv", index=False)

    print(f"✅ DONE: Association Mining completed!")
    print(f"-> Found {len(rules)} rules and {len(frequent_itemsets)} itemsets.")
    print(f"-> Artifacts saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()