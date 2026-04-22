import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD DATA
# ==============================
print("Loading data...")
df = pd.read_csv(r"D:\Smart-Movie-Recommender\data\cleaned\integrated_dataset_encoded.csv")

# ==============================
# MAP
# ==============================
user_map = df[['user_idx', 'user_id']].drop_duplicates().set_index('user_idx')
movie_map = df[['movie_idx', 'movie_title']].drop_duplicates().set_index('movie_idx')

# ==============================
# SPLIT PER USER (GIỮ USER ÍT DATA)
# ==============================
print("Splitting train/test (per user)...")

train_list = []
test_list = []

for user, group in df.groupby('user_idx'):

    if len(group) < 3:   # 👈 giữ user ít data
        train_list.append(group)
        continue

    group = group.sample(frac=1, random_state=42)

    test = group.head(1)
    train = group.iloc[1:]

    train_list.append(train)
    test_list.append(test)

train_df = pd.concat(train_list)
test_df = pd.concat(test_list) if len(test_list) > 0 else pd.DataFrame()

# ==============================
# USER-ITEM MATRIX
# ==============================
print("Building user-item matrix...")
user_item = train_df.pivot_table(
    index='user_idx',
    columns='movie_idx',
    values='rating'
).fillna(0)

# ==============================
# SVD (CF)
# ==============================
print("Training SVD...")
svd = TruncatedSVD(n_components=50, random_state=42)

user_factors = svd.fit_transform(user_item)
item_factors = svd.components_

mf_pred = np.dot(user_factors, item_factors)

mf_df = pd.DataFrame(
    mf_pred,
    index=user_item.index,
    columns=user_item.columns
)

# ==============================
# CONTENT-BASED
# ==============================
print("Building content features...")

content_features = train_df.drop_duplicates('movie_idx').set_index('movie_idx')

feature_cols = [col for col in train_df.columns if col.startswith('genre_')] + [
    'year_scaled',
    'genre_count_scaled',
    'primary_genre_encoded'
]

content_matrix = content_features[feature_cols]

content_sim = cosine_similarity(content_matrix)

content_sim_df = pd.DataFrame(
    content_sim,
    index=content_matrix.index,
    columns=content_matrix.index
)

# ==============================
# NORMALIZE
# ==============================
def normalize(scores):
    if scores.max() != scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min())
    return scores

# ==============================
# POPULAR (COLD START)
# ==============================
def get_popular(k=10):
    pop = train_df.groupby('movie_idx')['rating'].mean()
    return pop.sort_values(ascending=False).head(k)

# ==============================
# RECOMMEND HYBRID (FIX COLD START)
# ==============================
def recommend_hybrid(user, k=10, alpha=0.8):

    # user chưa tồn tại
    if user not in user_item.index:
        return get_popular(k)

    watched = user_item.loc[user]
    watched_items = watched[watched > 0].index

    # ❗ user xem 1–2 phim → CONTENT
    if len(watched_items) <= 2:
        scores = content_sim_df[watched_items].mean(axis=1)
        scores = normalize(scores)

        scores = scores.drop(watched_items, errors='ignore')

        return scores.sort_values(ascending=False).head(k)

    # HYBRID
    cf_scores = mf_df.loc[user]
    content_scores = content_sim_df[watched_items].mean(axis=1)

    cf_scores = normalize(cf_scores)
    content_scores = normalize(content_scores)

    content_scores = content_scores * 1.1

    final = alpha * cf_scores + (1 - alpha) * content_scores
    final = normalize(final)

    final = final.drop(watched_items, errors='ignore')

    return final.sort_values(ascending=False).head(k)

# ==============================
# EVALUATION
# ==============================
def precision_at_k(model, k=10, threshold=4):
    scores = []

    for user in test_df['user_idx'].unique():
        user_test = test_df[test_df['user_idx'] == user]

        actual = user_test[user_test['rating'] >= threshold]['movie_idx'].values
        if len(actual) == 0:
            continue

        recs = model(user, k)
        pred = recs.index.values

        hits = len(set(pred) & set(actual))
        scores.append(hits / k)

    return np.mean(scores) if scores else 0


def recall_at_k(model, k=10, threshold=4):
    scores = []

    for user in test_df['user_idx'].unique():
        user_test = test_df[test_df['user_idx'] == user]

        actual = user_test[user_test['rating'] >= threshold]['movie_idx'].values
        if len(actual) == 0:
            continue

        recs = model(user, k)
        pred = recs.index.values

        hits = len(set(pred) & set(actual))
        scores.append(hits / len(actual))

    return np.mean(scores) if scores else 0

# ==============================
# TUNING ALPHA
# ==============================
print("\n===== TUNING ALPHA =====")

best_alpha = 0
best_score = 0

for a in np.arange(0.6, 0.96, 0.05):
    score = precision_at_k(lambda u, k: recommend_hybrid(u, k, alpha=a))
    print(f"alpha={a:.2f} -> Precision={score:.5f}")

    if score > best_score:
        best_score = score
        best_alpha = a

print(f"\nBEST ALPHA: {best_alpha} | Precision: {best_score}")

# ==============================
# FINAL EVALUATION
# ==============================
print("\n===== MODEL EVALUATION =====")

print("Hybrid Precision@10:", precision_at_k(lambda u, k: recommend_hybrid(u, k, best_alpha)))
print("Hybrid Recall@10:", recall_at_k(lambda u, k: recommend_hybrid(u, k, best_alpha)))

# ==============================
# EXPORT CSV (FINAL)
# ==============================
print("\n===== EXPORTING CSV =====")

rows = []

for user in user_item.index:
    recs = recommend_hybrid(user, k=10, alpha=best_alpha)
    user_name = user_map['user_id'].get(user, "Unknown")

    for rank, (movie, score) in enumerate(recs.items(), 1):
        movie_title = movie_map['movie_title'].get(movie, "Unknown")

        rows.append({
            "user_id": user_name,
            "movie_id": movie,
            "movie_title": movie_title,
            "rank": rank,
            "score": round(float(score), 3)
        })

result_df = pd.DataFrame(rows)
result_df = result_df.sort_values(by=["user_id", "rank"])

result_df.to_csv("D:\Smart-Movie-Recommender\data\cleaned/recommendations.csv",
                 index=False,
                 encoding='utf-8-sig')
print("\nDONE -> hybrid_recommendations_final.csv")