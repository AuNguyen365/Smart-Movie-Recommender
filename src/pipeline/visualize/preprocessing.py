import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_genre_count_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """a) Phân bố số lượng thể loại phim (Genre Count Distribution)"""
    plt.figure(figsize=(10, 6))
    if 'genre_count' in df.columns:
        sns.countplot(data=df, x='genre_count', hue='genre_count', palette='magma', legend=False)
        plt.title('Genre Count Distribution')
        plt.xlabel('Number of Genres')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, "genre_count column not found", ha="center", va="center")
        
    plt.tight_layout()
    out_path = out_dir / "genre_count_distribution.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path

def plot_rating_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """b) Phân bố rating (Rating Distribution)"""
    plt.figure(figsize=(10, 6))
    if 'rating' in df.columns:
        sns.histplot(df['rating'], bins=10, kde=True, color='royalblue')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, "rating column not found", ha="center", va="center")
        
    plt.tight_layout()
    out_path = out_dir / "rating_distribution.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path

def plot_source_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """c) Phân bố nguồn dữ liệu (Source Distribution)"""
    plt.figure(figsize=(10, 6))
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Source Data Distribution')
    else:
        plt.text(0.5, 0.5, "source column not found", ha="center", va="center")
        
    plt.tight_layout()
    out_path = out_dir / "source_distribution.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path

def plot_genre_after_preprocessing(df: pd.DataFrame, out_dir: Path) -> Path:
    """d) Phân bố thể loại sau tiền xử lý (Genre Distribution After Preprocessing)"""
    genre_cols = [col for col in df.columns if col.startswith('genre_') and not col.endswith('_scaled') and col != 'genre_count']
    
    plt.figure(figsize=(12, 8))
    if genre_cols:
        genre_sums = df[genre_cols].sum().sort_values(ascending=False)
        sns.barplot(x=genre_sums.values, y=genre_sums.index, hue=genre_sums.index, palette='viridis', legend=False)
        plt.title('Genre Distribution After Preprocessing (Encoded)')
        plt.xlabel('Count')
        plt.ylabel('Genre')
    else:
        plt.text(0.5, 0.5, "No encoded genre columns found", ha="center", va="center")
        
    plt.tight_layout()
    out_path = out_dir / "genre_distribution_after_preprocessing.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path
