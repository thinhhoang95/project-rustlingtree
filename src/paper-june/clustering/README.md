# Clustering Ablation

Run the reproducible clustering ablation against the April 1 PPE ground truth:

```bash
python src/paper-june/clustering/ablation.py \
  --ground-truth data/artifacts/ppe/2026-04-01/ground_truth
```

By default the script infers the source PPE run from the ground-truth manifest,
loads its resampled tracks and feature matrix, sweeps KMeans over the configured
K range, evaluates silhouette, Calinski-Harabasz, Davies-Bouldin, and an inertia
elbow, then matches discovered medoids to ground-truth subcluster medoids.

Default output:

```text
data/artifacts/ppe/2026-04-01/paper-june/clustering-ablation/
```

Important files:

- `findings.md`: human-readable summary and interpretation.
- `validity_metrics.csv`: cluster validity metrics by K.
- `match_scores.csv`: all-cluster and pruned-cluster medoid scores by K.
- `figures/validity_metrics.png`: metric curves.
- `figures/contact_sheets/*.png`: cluster contact sheets with medoid overlays.
