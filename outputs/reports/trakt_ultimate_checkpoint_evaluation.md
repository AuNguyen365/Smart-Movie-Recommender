# Dataset Evaluation: trakt_ultimate_checkpoint

## Source
- source file: `data/trakt_ultimate_checkpoint.csv`
- cleaned file: `data/cleaned/trakt_ultimate_checkpoint_cleaned.csv`

## Size Snapshot
- rows before cleaning: 11500
- rows after cleaning: 5251
- columns: 13

## Cleaning Actions
- exact duplicate rows removed: 4850
- rows removed due to missing core fields (`user_id`, `movie_id`, `rating`): 0

## Missing Data (Top Fields)
- `cast`: 2.17%
- `genres`: 0.72%

## Rating Balance Check
- negative: 1157
- neutral: 2012
- positive: 2082
- min/max class ratio: 0.556
- verdict: **moderately-imbalanced**

## Visualizations
- ![](../figures/trakt_ultimate_checkpoint/missing_values.png)
- ![](../figures/trakt_ultimate_checkpoint/rating_distribution.png)
- ![](../figures/trakt_ultimate_checkpoint/top_genres.png)
- ![](../figures/trakt_ultimate_checkpoint/language_distribution.png)
- ![](../figures/trakt_ultimate_checkpoint/release_year_distribution.png)

## Recommendation
Dataset is usable, but classification tasks should consider class weights or targeted resampling.
