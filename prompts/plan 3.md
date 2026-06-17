Could you help me plan for a batch of experiments for a new conference paper based on the PPE framework. All baselines and analysis scripts should be implemented in `src/paper-june`:

# Background
The key premise is that a VLM, with its vast internal knowledge, can outperform many clustering techniques which is clumsy, signal-specific, and requires a lot of parameter tuning. VLM can understand the context (like air traffic management), and when placed in an agent orchestration environment, capable of outperforming many vanilla (not heavily tuned) classical approaches, and can even work with more sophisticated structure.

In order to show the value of our PPE approach, we will implement a set of baselines, each will serve an elaborated narrative.

# Narrative 2: Intervention Window Detection

An intervention window is where path stretching (like dogleg or trombone patterns) appear where controllers attempt to create distance to prevent conflicts. We will use peak detection algorithm (from scipy) and the residual energy curve in order to detect intervention windows, and compare against the PPE detected intervention windows.

The Intersection of Union (already reported in PPE evaluation suite) should be used for the baselines as well, and all results (baselines, PPE evals) should be centralized into a single Jupyter notebook like `/Volumes/CrucialX/project-rustlingtree/data/artifacts/ppe/2026-04-01/paper-june/clustering-ablation/ppe_vs_classical_comparison.ipynb`. 

Use 0.1 as the threshold, and make sure the code stay consistent between evaluation and baselines to prevent drifts. 


### Remarks
- For IoU and thresholding, we will assume class-agnostics.
- Report the classification performance separately in the table.
