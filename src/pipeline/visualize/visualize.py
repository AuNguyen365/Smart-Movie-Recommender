from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================
# PATH SETUP
# =====================================
BASE_DIR = Path(__file__).resolve().parents[3]

RULE_DIR = BASE_DIR / "outputs" / "association"
CLUSTER_DIR = BASE_DIR / "outputs" / "clustering"
REC_DIR = BASE_DIR / "outputs" / "recommendation"

# save charts
CHART_DIR = BASE_DIR / "outputs" / "figures"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# LOAD FILES
# =====================================
rules = pd.read_csv(RULE_DIR / "rules.csv")
users_clustered = pd.read_parquet(CLUSTER_DIR / "labels.parquet")
cluster_profiles = pd.read_csv(CLUSTER_DIR / "user_cluster_profiles.csv")
recommendations = pd.read_csv(REC_DIR / "recommendations.csv")

# =====================================
# STYLE
# =====================================
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# =====================================
# AUTO DETECT COLUMN
# =====================================
def find_cluster_column(df):
    keywords = ["cluster", "label", "segment", "group"]

    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col

    for col in df.columns:
        if df[col].nunique() <= 10:
            return col

    return df.columns[-1]


def find_movie_column(df):
    keywords = ["movie", "title", "film", "name"]

    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col

    return df.columns[0]


cluster_col = find_cluster_column(users_clustered)
profile_cluster_col = find_cluster_column(cluster_profiles)
movie_col = find_movie_column(recommendations)

print("Detected users cluster column:", cluster_col)
print("Detected profile cluster column:", profile_cluster_col)
print("Detected recommendation movie column:", movie_col)

# =====================================
# SAVE FUNCTION
# =====================================
def save_chart(filename):
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

# =====================================
# 1. USERS PER CLUSTER
# =====================================
plt.figure()
users_clustered[cluster_col].value_counts().sort_index().plot(kind="bar")
plt.title("Number of Users in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Users")
save_chart("01_users_per_cluster.png")

# =====================================
# 2. TOP 10 RULES BY LIFT
# =====================================
if "lift" in rules.columns:
    top_rules = rules.sort_values("lift", ascending=False).head(10)

    label_col = "antecedents_str"
    if label_col not in rules.columns:
        label_col = rules.columns[0]

    plt.figure(figsize=(11, 6))
    plt.barh(top_rules[label_col].astype(str), top_rules["lift"])
    plt.title("Top 10 Association Rules by Lift")
    plt.xlabel("Lift")
    plt.gca().invert_yaxis()
    save_chart("02_top_rules_lift.png")

# =====================================
# 3. SUPPORT VS CONFIDENCE
# =====================================
if "support" in rules.columns and "confidence" in rules.columns:
    plt.figure()
    plt.scatter(rules["support"], rules["confidence"], alpha=0.7)
    plt.title("Support vs Confidence")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    save_chart("03_support_confidence.png")

# =====================================
# 4. TOP RECOMMENDED MOVIES
# =====================================
movie_counts = recommendations[movie_col].astype(str).value_counts().head(10)

plt.figure(figsize=(11, 6))
movie_counts.plot(kind="bar")
plt.title("Top 10 Most Recommended Movies")
plt.xlabel("Movie")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
save_chart("04_top_recommended_movies.png")

# =====================================
# 5. CLUSTER PROFILE HEATMAP
# =====================================
profile_data = cluster_profiles.copy()

if profile_cluster_col in profile_data.columns:
    profile_data = profile_data.set_index(profile_cluster_col)

numeric_data = profile_data.select_dtypes(include=[np.number])

if not numeric_data.empty:
    plt.figure(figsize=(12, 6))
    plt.imshow(numeric_data, aspect="auto")
    plt.colorbar()
    plt.title("Cluster Profile Heatmap")
    plt.yticks(range(len(numeric_data.index)), numeric_data.index)
    plt.xticks(
        range(len(numeric_data.columns)),
        numeric_data.columns,
        rotation=90
    )
    save_chart("05_cluster_heatmap.png")

# =====================================
# 6. DISTRIBUTION OF LIFT
# =====================================
if "lift" in rules.columns:
    plt.figure()
    rules["lift"].hist(bins=20)
    plt.title("Distribution of Lift Values")
    plt.xlabel("Lift")
    plt.ylabel("Frequency")
    save_chart("06_lift_distribution.png")

print("\nAll charts saved successfully!")
print("Folder:", CHART_DIR)