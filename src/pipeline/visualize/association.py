import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_rules_scatter(rules_df: pd.DataFrame, out_dir: Path) -> Path:
    """Vẽ biểu đồ phân tán Support vs Confidence, màu sắc theo Lift."""
    plt.figure(figsize=(10, 6))
    
    if rules_df.empty:
        plt.text(0.5, 0.5, "No rules found", ha="center", va="center")
    else:
        scatter = plt.scatter(
            rules_df['support'], 
            rules_df['confidence'], 
            c=rules_df['lift'], 
            cmap='viridis', 
            alpha=0.6
        )
        plt.colorbar(scatter, label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Association Rules: Support vs Confidence')
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    out_path = out_dir / "association_rules_scatter.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path

def plot_top_rules_by_lift(rules_df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> Path:
    """Vẽ biểu đồ cột cho Top N luật kết hợp theo Lift."""
    plt.figure(figsize=(12, 8))
    
    if rules_df.empty:
        plt.text(0.5, 0.5, "No rules found", ha="center", va="center")
    else:
        # Tạo nhãn cho luật (Antecedents -> Consequents)
        top_rules = rules_df.sort_values('lift', ascending=False).head(top_n).copy()
        
        # Xử lý chuỗi nếu antecedents_str/consequents_str là list
        top_rules['rule_label'] = top_rules.apply(
            lambda x: f"{x['antecedents_str']} -> {x['consequents_str']}", axis=1
        )
        
        sns.barplot(data=top_rules, x='lift', y='rule_label', hue='lift', palette='rocket', legend=False)
        plt.title(f'Top {top_n} Association Rules by Lift')
        plt.xlabel('Lift')
        plt.ylabel('Rule')

    plt.tight_layout()
    out_path = out_dir / "top_association_rules_lift.png"
    plt.savefig(out_path, dpi=140)
    plt.close()
    return out_path
