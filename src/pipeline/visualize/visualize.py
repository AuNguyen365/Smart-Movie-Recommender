from pathlib import Path
import pandas as pd
import json

# =====================================
# PATH
# =====================================
BASE_DIR = Path(__file__).resolve().parents[3]

RULE_DIR = BASE_DIR / "outputs" / "association"
CLUSTER_DIR = BASE_DIR / "outputs" / "clustering"
REC_DIR = BASE_DIR / "outputs" / "recommendation"

TOP_N = 10

# =====================================
# LOAD DATA
# =====================================
rules = pd.read_csv(RULE_DIR / "rules.csv")
users = pd.read_parquet(CLUSTER_DIR / "labels.parquet")
profiles = pd.read_csv(CLUSTER_DIR / "user_cluster_profiles.csv")
recommendations = pd.read_csv(REC_DIR / "recommendations.csv")

# =====================================
# LOAD MOVIE DICTIONARY
# =====================================
movie_map = {}
dict_file = RULE_DIR / "dictionary.json"

if dict_file.exists():
    with open(dict_file, "r", encoding="utf-8") as f:
        movie_map = json.load(f)

# =====================================
# HELPERS
# =====================================
def find_cluster_col(df):
    for col in df.columns:
        if "cluster" in col.lower() or "label" in col.lower():
            return col
    return df.columns[-1]

def clean_text(text):
    return str(text).replace("[", "").replace("]", "").replace("'", "")

cluster_col = find_cluster_col(users)

# =====================================
# HEADER
# =====================================
print("=" * 60)
print("SMART MOVIE RECOMMENDER REPORT".center(60))
print("=" * 60)

# =====================================
# OVERVIEW
# =====================================
print("\n[OVERVIEW]")
print("Users:", len(users))
print("Rules:", len(rules))
print("Clusters:", users[cluster_col].nunique())
print("Recommendations:", len(recommendations))

# =====================================
# ASSOCIATION RULES
# =====================================
print("\n[ASSOCIATION RULES - TOP 10]")

# Sort tốt hơn (không đổi data, chỉ đổi cách nhìn)
rules_sorted = rules.sort_values(
    ["lift", "confidence", "support"],
    ascending=False
)

top_rules = rules_sorted.head(TOP_N)

for i, (_, row) in enumerate(top_rules.iterrows(), 1):
    ant = clean_text(row["antecedents_str"])
    con = clean_text(row["consequents_str"])
    print(f"{i}. {ant} -> {con}")

print("\n[ASSOCIATION DETAIL]")
print(top_rules[[
    "antecedents_str",
    "consequents_str",
    "support",
    "confidence",
    "lift"
]])

# =====================================
# CLUSTERING
# =====================================
print("\n[CLUSTER DISTRIBUTION]")

cluster_counts = users[cluster_col].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} users")

print("\n[CLUSTER INSIGHT]")
for _, row in profiles.head(TOP_N).iterrows():
    cid = row["cluster_id"]
    g1 = row["top_genre_1"]
    g2 = row["top_genre_2"]
    print(f"Cluster {cid}: prefers {g1} / {g2}")

print("\n[CLUSTER SAMPLE USERS]")
sample_cluster = users[cluster_col].iloc[0]
print(users[users[cluster_col] == sample_cluster].head())

# =====================================
# RECOMMENDATION
# =====================================
print("\n[RECOMMENDATION OVERVIEW]")
print("Total users (in recommendation):", recommendations["user_id"].nunique())
print("Total movies:", recommendations["movie_id"].nunique())

# TOP MOVIES
print("\n[TOP MOVIES FROM RECOMMENDATION]")

movies = recommendations["movie_title"].astype(str)

if movie_map:
    movies = movies.map(lambda x: movie_map.get(str(x), x))

top_movies = movies.value_counts().head(TOP_N)

for i, (movie, count) in enumerate(top_movies.items(), 1):
    print(f"{i}. {movie} ({count})")

# SAMPLE USER
print("\n[SAMPLE RECOMMENDATION PER USER]")

sample_user = recommendations["user_id"].iloc[0]
user_rec = recommendations[recommendations["user_id"] == sample_user]

print(f"User: {sample_user}")
print(user_rec.sort_values("rank"))

# GLOBAL TOP SCORE
print("\n[TOP SCORE GLOBAL]")
print(recommendations.sort_values(by="score", ascending=False).head(10))

# SCORE STATS
print("\n[SCORE STATISTICS]")
print(recommendations["score"].describe())

# =====================================
# FOOTER
# =====================================
print("\n" + "=" * 60)
print("END OF REPORT".center(60))
print("=" * 60)