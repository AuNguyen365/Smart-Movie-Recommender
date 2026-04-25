import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_score_distribution(recs_df: pd.DataFrame, out_dir: Path) -> Path:
    """Vẽ biểu đồ phân phối điểm gợi ý (scores)."""
    plt.figure(figsize=(10, 6))
    
    if recs_df.empty:
        plt.text(0.5, 0.5, "No recommendations found", ha="center", va="center")
    else:
        sns.histplot(recs_df['score'], bins=30, kde=True, color='skyblue')
        plt.title('Distribution of Recommendation Scores')
        plt.xlabel('Score')
        plt.ylabel('Frequency')

    plt.tight_layout()
    out_path = out_dir / "recommendation_score_dist.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path

def plot_popular_recommendations(recs_df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> Path:
    """Vẽ biểu đồ các bộ phim được gợi ý nhiều nhất (tần suất xuất hiện trong Top-N)."""
    plt.figure(figsize=(12, 8))
    
    if recs_df.empty:
        plt.text(0.5, 0.5, "No recommendations found", ha="center", va="center")
    else:
        pop_recs = recs_df['movie_title'].value_counts().head(top_n).reset_index()
        pop_recs.columns = ['movie_title', 'count']
        
        sns.barplot(data=pop_recs, x='count', y='movie_title', hue='count', palette='viridis', legend=False)
        plt.title(f'Top {top_n} Most Frequently Recommended Movies')
        plt.xlabel('Recommendation Count')
        plt.ylabel('Movie Title')

    plt.tight_layout()
    out_path = out_dir / "popular_recommendations.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path

def plot_rank_vs_score(recs_df: pd.DataFrame, out_dir: Path) -> Path:
    """Vẽ biểu đồ Rank vs Score để kiểm tra tính nhất quán."""
    plt.figure(figsize=(10, 6))
    
    if recs_df.empty:
        plt.text(0.5, 0.5, "No recommendations found", ha="center", va="center")
    else:
        sns.boxplot(data=recs_df, x='rank', y='score', hue='rank', palette='Blues', legend=False)
        plt.title('Recommendation Score by Rank')
        plt.xlabel('Rank')
        plt.ylabel('Score')

    plt.tight_layout()
    out_path = out_dir / "recommendation_rank_vs_score.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path
