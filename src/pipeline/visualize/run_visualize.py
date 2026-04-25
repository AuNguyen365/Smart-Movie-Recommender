import pandas as pd
from pathlib import Path
from src.pipeline.visualize.association import plot_rules_scatter, plot_top_rules_by_lift
from src.pipeline.visualize.recommendation import plot_score_distribution, plot_popular_recommendations, plot_rank_vs_score
from src.pipeline.visualize.preprocessing import (
    plot_genre_count_distribution, 
    plot_rating_distribution, 
    plot_source_distribution, 
    plot_genre_after_preprocessing
)

def main():
    # Paths
    BASE_DIR = Path(__file__).resolve().parents[3]
    OUTPUTS_DIR = BASE_DIR / "outputs"
    FIGURES_DIR = OUTPUTS_DIR / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Visualize Preprocessing (Encoded Data)
    encoding_path = OUTPUTS_DIR / "data_split" / "train.csv"
    if encoding_path.exists():
        print(f"Loading encoded data from {encoding_path}...")
        df = pd.read_csv(encoding_path)
        pre_dir = FIGURES_DIR / "preprocessing"
        
        # Xóa các file cũ trong thư mục preprocessing nếu có
        if pre_dir.exists():
            for f in pre_dir.glob("*.png"):
                f.unlink()
        pre_dir.mkdir(parents=True, exist_ok=True)
        
        plot_genre_count_distribution(df, pre_dir)
        plot_rating_distribution(df, pre_dir)
        plot_source_distribution(df, pre_dir)
        plot_genre_after_preprocessing(df, pre_dir)
        print(f"Preprocessing visualizations saved to {pre_dir}")
    else:
        print("Encoded data file (train.csv) not found. Skipping preprocessing visualization.")

    # 2. Visualize Association Rules
    rules_path = OUTPUTS_DIR / "association" / "rules.csv"
    if rules_path.exists():
        print(f"Loading rules from {rules_path}...")
        rules_df = pd.read_csv(rules_path)
        assoc_dir = FIGURES_DIR / "association"
        assoc_dir.mkdir(parents=True, exist_ok=True)
        
        plot_rules_scatter(rules_df, assoc_dir)
        plot_top_rules_by_lift(rules_df, assoc_dir)
        print(f"Association visualizations saved to {assoc_dir}")
    else:
        print("Rules file not found. Skipping association visualization.")

    # 3. Visualize Recommendations
    recs_path = OUTPUTS_DIR / "recommendation" / "recommendations.csv"
    if recs_path.exists():
        print(f"Loading recommendations from {recs_path}...")
        recs_df = pd.read_csv(recs_path)
        rec_dir = FIGURES_DIR / "recommendation"
        rec_dir.mkdir(parents=True, exist_ok=True)
        
        plot_score_distribution(recs_df, rec_dir)
        plot_popular_recommendations(recs_df, rec_dir)
        plot_rank_vs_score(recs_df, rec_dir)
        print(f"Recommendation visualizations saved to {rec_dir}")
    else:
        print("Recommendations file not found. Skipping recommendation visualization.")

if __name__ == "__main__":
    main()
