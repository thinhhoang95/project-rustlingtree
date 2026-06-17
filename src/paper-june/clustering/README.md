# Clustering Ablation

Run the reproducible clustering ablation against the April 1 PPE ground truth:

```bash
python src/paper-june/clustering/ablation.py \
  --ground-truth data/artifacts/ppe/2026-04-01/ground_truth
```

By default the script infers the source PPE run from the ground-truth manifest,
loads its resampled tracks and feature matrix, sweeps KMeans over the configured
K range, evaluates silhouette, Calinski-Harabasz, Davies-Bouldin, and an inertia
elbow, then matches discovered medoids to ground-truth subcluster medoids. It
also sweeps DBSCAN, HDBSCAN, and Leiden community detection, ranking the top two
parameter sets for each by pruned medoid-matching F1.

Default output:

```text
data/artifacts/ppe/2026-04-01/paper-june/clustering-ablation/
```

Important files:

- `findings.md`: human-readable summary and interpretation.
- `validity_metrics.csv`: cluster validity metrics by K.
- `match_scores.csv`: all-cluster and pruned-cluster medoid scores by K.
- `density_sweep_metrics.csv`: intrinsic metrics for every DBSCAN/HDBSCAN/Leiden parameter set.
- `density_match_scores.csv`: medoid scores for every DBSCAN/HDBSCAN/Leiden parameter set.
- `selected_density_scores.csv`: top two selected parameter sets per density/community algorithm.
- `figures/validity_metrics.png`: metric curves.
- `figures/contact_sheets/*.png`: cluster contact sheets with medoid overlays.
