import ast
import json
import re
from pathlib import Path

import pandas as pd


def parse_genres(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip().lower() for x in value if str(x).strip()]

    raw = str(value).strip()
    if not raw:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, list):
                out = []
                for item in parsed:
                    if isinstance(item, dict) and item.get('name'):
                        out.append(str(item['name']).strip().lower())
                    elif isinstance(item, str):
                        txt = item.strip().lower()
                        if txt:
                            out.append(txt)
                if out:
                    return out
        except Exception:
            pass

    text = raw.strip('[]')
    parts = [p.strip().strip('"\'') for p in re.split(r'[,|;/]', text)]
    return [p.lower() for p in parts if p]


path = Path('data/cleaned/integrated_dataset_cleaned.csv')
df = pd.read_csv(path)

# 1) Normalize timestamp into a single ISO format with UTC suffix.
ts = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed', utc=True)
if ts.isna().any():
    raise ValueError(f'Found unparsable timestamp rows: {int(ts.isna().sum())}')
# Keep milliseconds to balance compactness and precision.
df['timestamp'] = ts.dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str.slice(0, 23) + 'Z'

# 2) Fill missing cast/genres by movie_id then movie_title fallback, then unknown.
for col in ['cast', 'genres']:
    if col not in df.columns:
        continue

    s = df[col].astype('string')
    missing_mask = s.isna() | s.str.strip().eq('')

    if 'movie_id' in df.columns:
        by_movie_id = (
            df.loc[~missing_mask, ['movie_id', col]]
            .dropna(subset=[col])
            .assign(_clean=lambda x: x[col].astype(str).str.strip())
        )
        by_movie_id = by_movie_id[by_movie_id['_clean'] != '']
        if not by_movie_id.empty:
            mode_movie = by_movie_id.groupby('movie_id')['_clean'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
            fill_vals = df['movie_id'].map(mode_movie)
            s = s.mask(missing_mask, fill_vals)
            missing_mask = s.isna() | s.str.strip().eq('')

    if 'movie_title' in df.columns and missing_mask.any():
        by_title = (
            df.loc[~missing_mask, ['movie_title', col]]
            .dropna(subset=[col])
            .assign(_clean=lambda x: x[col].astype(str).str.strip())
        )
        by_title = by_title[by_title['_clean'] != '']
        if not by_title.empty:
            mode_title = by_title.groupby('movie_title')['_clean'].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
            fill_vals = df['movie_title'].map(mode_title)
            s = s.mask(missing_mask, fill_vals)
            missing_mask = s.isna() | s.str.strip().eq('')

    s = s.fillna('unknown')
    s = s.mask(s.str.strip().eq(''), 'unknown')
    df[col] = s.astype(str)

# 3) Rebuild genre-derived fields to keep consistency.
if 'genres' in df.columns:
    genres_list = df['genres'].apply(parse_genres)
    df['genres_list'] = genres_list.apply(lambda g: json.dumps(g, ensure_ascii=True))
    df['primary_genre'] = genres_list.apply(lambda g: g[0] if g else 'unknown')
    df['genre_count'] = genres_list.apply(len)

# Save in place.
df.to_csv(path, index=False)

# Print summary for quick verification.
missing_pct = (df[['cast', 'genres']].isna().mean() * 100).to_dict() if {'cast', 'genres'}.issubset(df.columns) else {}
empty_cast = int((df['cast'].astype(str).str.strip() == '').sum()) if 'cast' in df.columns else -1
empty_genres = int((df['genres'].astype(str).str.strip() == '').sum()) if 'genres' in df.columns else -1
print('rows', len(df))
print('timestamp_sample', df['timestamp'].head(3).tolist())
print('missing_pct_cast_genres', missing_pct)
print('empty_cast', empty_cast)
print('empty_genres', empty_genres)
print('unknown_cast', int((df['cast'].astype(str).str.lower() == 'unknown').sum()) if 'cast' in df.columns else -1)
print('unknown_genres', int((df['genres'].astype(str).str.lower() == 'unknown').sum()) if 'genres' in df.columns else -1)
