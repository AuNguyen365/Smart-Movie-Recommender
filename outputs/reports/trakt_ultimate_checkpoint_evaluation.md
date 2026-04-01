# Dataset Evaluation: trakt_ultimate_checkpoint

## Source
- source file: `data/trakt_ultimate_checkpoint.csv`
- cleaned file: `data/cleaned/trakt_ultimate_checkpoint_cleaned.csv`

## Size Snapshot
- rows before cleaning: 10000
- rows after cleaning: 5150
- columns: 13

## Cleaning Actions
- exact duplicate rows removed: 4850
- rows removed due to missing core fields (`user_id`, `movie_id`, `rating`): 0

## Missing Data (Top Fields)
- `cast`: 3.81%
- `genres`: 0.80%

## Rating Balance Check
- negative: 765
- neutral: 1856
- positive: 2529
- min/max class ratio: 0.302
- verdict: **imbalanced**

## Visualizations
- ![](../figures/trakt_ultimate_checkpoint/missing_values.png)
- ![](../figures/trakt_ultimate_checkpoint/rating_distribution.png)
- ![](../figures/trakt_ultimate_checkpoint/top_genres.png)
- ![](../figures/trakt_ultimate_checkpoint/language_distribution.png)
- ![](../figures/trakt_ultimate_checkpoint/release_year_distribution.png)

## Recommendation
Dataset is imbalanced for rating-class tasks; apply resampling or class-weighted training if you build a classifier.
