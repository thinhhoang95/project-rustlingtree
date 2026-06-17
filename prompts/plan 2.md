Could you help me plan for a batch of experiments for a new conference paper based on the PPE framework. All baselines and analysis scripts should be implemented in `src/paper-june`:

# Background
The key premise is that a VLM, with its vast internal knowledge, can outperform many clustering techniques which is clumsy, signal-specific, and requires a lot of parameter tuning. VLM can understand the context (like air traffic management), and when placed in an agent orchestration environment, capable of outperforming many vanilla (not heavily tuned) classical approaches, and can even work with more sophisticated structure.

In order to show the value of our PPE approach, we will implement a set of baselines, each will serve an elaborated narrative.

# Narrative 1: Clustering
It is known in the literature that no universal cluster validity indices set are available, to determine the number of clusters. Let's use silhouette, Calkinski, Davies... for choosing k (i.e., the standard techniques), and report your findings about whether you could identify the same number of subclusters from the ground truth. I think you would expect that it is impossible to yield the exact subclusters identified, even if one is allowed to discard "noisy" clusters.

Render a contact sheet of identified clusters and medoid paths representing each cluster. 

The entire experiment should be implemented in code to be reproducible in the future. The final results should be rendered to a subdirectory so they can be reviewed manually in the future.  The final findings are reported in a markdown file in that folder too.

Code to be implemented in `src/paper-june/clustering`.

## Narrative 1.2: Density-based Clustering



### Remarks
- The identified clusters should directly compare to the ground-truth subclusters, not the clusters.
