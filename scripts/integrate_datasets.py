import pandas as pd
from pathlib import Path

p1 = Path('data/cleaned/movie_final_dataset_cleaned.csv')
p2 = Path('data/cleaned/trakt_ultimate_checkpoint_cleaned.csv')
out = Path('data/cleaned/integrated_dataset_cleaned.csv')

d1 = pd.read_csv(p1)
d2 = pd.read_csv(p2)

d1['source'] = 'movie_final_dataset'
d2['source'] = 'trakt_ultimate_checkpoint'

merged = pd.concat([d1, d2], ignore_index=True)
before = len(merged)

merged = merged.drop_duplicates()
after_exact = len(merged)

key_cols = [c for c in ['user_id', 'movie_id', 'rating', 'timestamp'] if c in merged.columns]
if key_cols:
    merged = merged.drop_duplicates(subset=key_cols, keep='first')
after_key = len(merged)

out.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out, index=False)

print('movie_rows', len(d1))
print('trakt_rows', len(d2))
print('merged_before_dedup', before)
print('merged_after_exact_dedup', after_exact)
print('merged_after_key_dedup', after_key)
print('output', out.as_posix())
