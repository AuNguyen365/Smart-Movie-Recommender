import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import json
import matplotlib.pyplot as plt
import sys
import io

# Fix encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thiết lập đường dẫn
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "cleaned" / "integrated_dataset_encoded.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "data_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

def main():
    print("🚀 Starting Advanced Data Split (Group-based & 3-way)...")
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Group-based Split (Giữ User trong cùng một fold)
    # Chia 70% Train, 30% cho phần còn lại (Val + Test)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEED)
    train_idx, temp_idx = next(gss.split(df, groups=df['user_idx']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Chia 30% còn lại thành 50/50 -> 15% Val và 15% Test
    gss_val_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED)
    val_idx, test_idx = next(gss_val_test.split(temp_df, groups=temp_df['user_idx']))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    # 3. Lưu dữ liệu ra CSV
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # 4. Lưu Split Indices (Định dạng Parquet theo slide yêu cầu)
    indices_df = pd.DataFrame({
        "index": df.index,
        "split": "train"
    })
    indices_df.loc[val_df.index, "split"] = "val"
    indices_df.loc[test_df.index, "split"] = "test"
    
    indices_df.to_parquet(OUTPUT_DIR / "split_indices.parquet", engine='fastparquet', index=False)

    # 5. Sampling Report (Bổ sung Seed và tỷ lệ)
    report = {
        "metadata": {
            "seed": SEED,
            "split_strategy": "GroupShuffleSplit (User-based)",
            "target_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}
        },
        "stats": {
            "total_samples": len(df),
            "total_users": int(df['user_idx'].nunique()),
            "train": {"samples": len(train_df), "users": int(train_df['user_idx'].nunique()), "ratio": len(train_df)/len(df)},
            "val": {"samples": len(val_df), "users": int(val_df['user_idx'].nunique()), "ratio": len(val_df)/len(df)},
            "test": {"samples": len(test_df), "users": int(test_df['user_idx'].nunique()), "ratio": len(test_df)/len(df)}
        }
    }
    
    with open(OUTPUT_DIR / "sampling_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # 6. Visualization (Theo slide yêu cầu)
    plt.figure(figsize=(10, 6))
    labels = ['Training', 'Validation', 'Test']
    sizes = [len(train_df), len(val_df), len(test_df)]
    colors = ['#4f46e5', '#f59e0b', '#10b981']
    
    plt.bar(labels, sizes, color=colors)
    plt.title('Data Split Distribution (Sample Count)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples')
    
    # Thêm text hiển thị số lượng trên cột
    for i, v in enumerate(sizes):
        plt.text(i, v + 50, f"{v}\n({v/len(df)*100:.1f}%)", ha='center', fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "split_viz.png")
    plt.close()

    print(f"✅ DONE: Advanced Split completed!")
    print(f"-> Artifacts saved in: {OUTPUT_DIR}")
    print(f"-> Visualization saved as: split_viz.png")

if __name__ == "__main__":
    main()
