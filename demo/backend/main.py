import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import ast
import numpy as np
import re
from difflib import SequenceMatcher
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "outputs")), name="static")

REC_FILE = os.path.join(BASE_DIR, "outputs", "recommendation", "recommendations.csv")
RULES_FILE = os.path.join(BASE_DIR, "outputs", "association", "rules.csv")
METRICS_FILE = os.path.join(BASE_DIR, "outputs", "association", "raw_metrics.json")

# Global variables to hold data
recs_df = pd.DataFrame()
rules_df = pd.DataFrame()
full_rules_df = pd.DataFrame()
metrics_data = {}
popular_movies = []
cluster_profiles = []
user_to_cluster = {}

@app.on_event("startup")
def load_data():
    global recs_df, rules_df, full_rules_df, metrics_data, popular_movies, cluster_profiles, user_to_cluster
    try:
        recs_df = pd.read_csv(REC_FILE)
    except Exception as e:
        print(f"Error loading recommendations: {e}")
        
    try:
        full_df = pd.read_csv(RULES_FILE)
        import numpy as np
        full_df = full_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['lift'])
        full_rules_df = full_df
        rules_df = full_df.sort_values(by="lift", ascending=False).head(100)
    except Exception as e:
        print(f"Error loading rules: {e}")
        
    try:
        with open(METRICS_FILE, "r") as f:
            metrics_data = json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        
    try:
        if not recs_df.empty:
            pop = recs_df['movie_title'].value_counts().head(10).reset_index()
            pop.columns = ['movie_title', 'count']
            for idx, row in pop.iterrows():
                popular_movies.append({
                    "rank": idx + 1,
                    "movie_title": row['movie_title'],
                    "score": float(row['count'])
                })
    except Exception as e:
        pass
        
    try:
        CLUSTER_FILE = os.path.join(BASE_DIR, "outputs", "clustering", "user_cluster_profiles.parquet")
        cluster_df = pd.read_parquet(CLUSTER_FILE)
        for _, row in cluster_df.iterrows():
            top_g = []
            try:
                g_data = row['top_genres']
                if isinstance(g_data, str):
                    try:
                        g_list = json.loads(g_data)
                    except:
                        g_list = ast.literal_eval(g_data)
                elif isinstance(g_data, np.ndarray):
                    g_list = g_data.tolist()
                else:
                    g_list = g_data
                
                top_g = [g['genre'].replace('_', ' ').title() for g in g_list if isinstance(g, dict) and g.get('genre') not in ['count', 'count_scaled']]
            except Exception as ex:
                pass
                
            cluster_profiles.append({
                "cluster_id": int(row['cluster_id']),
                "num_users": int(row['num_users']),
                "avg_rating": float(row['avg_rating']),
                "top_genres": top_g[:3]
            })
    except Exception as e:
        print("Error loading clusters:", e)
        
    try:
        df_integ = pd.read_csv(os.path.join(BASE_DIR, "data", "cleaned", "integrated_dataset_encoded.csv"), usecols=['user_id', 'user_idx'])
        df_clust = pd.read_parquet(os.path.join(BASE_DIR, "outputs", "clustering", "users_clustered.parquet"), columns=['user_idx', 'user_cluster'])
        merged = df_integ.drop_duplicates(subset=['user_id']).merge(df_clust, on='user_idx')
        user_to_cluster = dict(zip(merged['user_id'], merged['user_cluster']))
    except Exception as e:
        print("Error mapping users to clusters", e)

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.get("/api/users")
def get_users():
    if recs_df.empty:
        return {"users": ["--- Người Dùng Mới (New User) ---"]}
    users = recs_df['user_id'].unique().tolist()
    users.insert(0, "--- Người Dùng Mới (New User) ---")
    return {"users": users}

@app.get("/api/recommendations/{user_id}")
def get_recommendations(user_id: str):
    if user_id == "--- Người Dùng Mới (New User) ---":
        return {"recommendations": popular_movies, "cluster_id": None}
    
    cluster_id = int(user_to_cluster[user_id]) if user_id in user_to_cluster else None
    
    if recs_df.empty:
        return {"recommendations": [], "cluster_id": cluster_id}
        
    user_recs = recs_df[recs_df['user_id'] == user_id].to_dict(orient="records")
    return {"recommendations": user_recs, "cluster_id": cluster_id}

@app.get("/api/rules")
def get_rules():
    if rules_df.empty:
        return {"rules": []}
    
    # We replace nan with None so it translates correctly to JSON null
    import math
    records = rules_df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return {"rules": records}

@app.get("/api/movie_recommendations")
def movie_recommendations(movie: str):
    if full_rules_df.empty or not movie:
        return {"recommendations": []}
    
    def is_match(antecedents_str, query):
        q_lower = query.lower()
        # Fast exact substring
        if q_lower in str(antecedents_str).lower():
            return True
            
        norm_query = re.sub(r'[^a-zA-Z0-9]', '', q_lower)
        q_words = re.findall(r'\w+', q_lower)
        
        try:
            movies = ast.literal_eval(antecedents_str)
        except:
            movies = [antecedents_str]
            
        for m in movies:
            m_lower = str(m).lower()
            norm_m = re.sub(r'[^a-zA-Z0-9]', '', m_lower)
            
            # Normalized substring (e.g. "ironman" in "ironman3")
            if norm_query and norm_query in norm_m:
                return True
                
            # Token match (e.g. "wick john" in "john wick")
            if q_words and all(w in m_lower for w in q_words):
                return True
                
            # Fuzzy match for slight typos (e.g. "jhon wick")
            if norm_query and len(norm_query) >= 4:
                if SequenceMatcher(None, norm_query, norm_m).ratio() > 0.75:
                    return True
                    
        return False

    mask = full_rules_df['antecedents_str'].apply(lambda x: is_match(x, movie))
    filtered = full_rules_df[mask]
    
    import ast
    recs = {}
    for _, row in filtered.iterrows():
        try:
            consequents_str = row['consequents_str']
            if pd.isna(consequents_str):
                continue
            
            # consequents_str is like "['Movie']"
            consequents = ast.literal_eval(consequents_str)
            for m in consequents:
                # Exclude the searched movie itself if it somehow appears in consequents
                if movie.lower() not in m.lower():
                    if m not in recs or row['lift'] > recs[m]['lift']:
                        recs[m] = {
                            "movie_title": m,
                            "confidence": row['confidence'],
                            "lift": row['lift']
                        }
        except Exception:
            pass
            
    sorted_recs = sorted(list(recs.values()), key=lambda x: x['lift'], reverse=True)
    return {"recommendations": sorted_recs[:20]}

@app.get("/api/metrics")
def get_metrics():
    return metrics_data

@app.get("/api/clusters")
def get_clusters():
    return {"clusters": cluster_profiles}
