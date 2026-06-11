Below is the implementation blueprint I would hand to the engineering team. The central recommendation is:

$$
\boxed{
\textbf{Use the VLM as a visual reviewer/orchestrator, not as the source of geometry.}
}
$$

The durable product of VLM-PPE should be a **typed, reproducible procedure program** extracted from ADS-B tracks by deterministic clustering, residual-window detection, and segmented-polyline optimization. This keeps the system agentic, but not fragile.

---

# 1. Refined research thesis

I would phrase the thesis as:

> **VLM-PPE is an ADS-B-first practical procedure extractor. It uses a VLM-guided agentic loop to inspect trajectory clusters and diagnostic plots, but all final procedure geometry is produced by deterministic tools: clustering, residual-energy analysis, fixed-token polyline fitting, and schema-validated export. The output is a practical procedure program that can be reconstructed by a client as a 2D path.**

This is stronger than “VLM parses charts.” AIPs are authoritative but not necessarily convenient as machine-readable operational usage data. ICAO describes AIP as the main publication of a State’s AIS office, while also noting the transition toward data-centric AIM products and services, so it is reasonable to treat AIP as an optional anchor/naming source rather than the primary inference engine. ([ICAO][1])

The conceptual core should stay aligned with what we discussed earlier: **model the edit, not the whole path**. A deviation program should contain $$s_{\text{in}}, s_{\text{out}}$$, a small number of straight vector legs, and a computed closure leg; the previous brainstorm explicitly motivated the closure leg because it removes degrees of freedom, guarantees rejoin-to-template, makes (G) continuous/derived, and exposes absorbed path length directly. 

---

# 2. Recommended system architecture

Use a **Python deterministic core** plus a **bounded agent graph**.

My recommended framework stack is:

| Layer               | Recommendation                                     | Reason                                                                                                                                                                                                                |
| ------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent orchestration | **LangGraph**                                      | Good fit for durable, stateful, human-in-the-loop agent workflows; its docs position it as a low-level runtime for long-running stateful agents with persistence and human-in-the-loop support. ([LangChain Docs][2]) |
| VLM                 | **Gemini 3.5 Flash**, pinned as `gemini-3.5-flash` | Current Gemini API docs list stable model names and give `gemini-3.5-flash` as a stable example; avoid `latest` aliases in research runs because they can be hot-swapped. ([Google AI for Developers][3])             |
| Data validation     | **Pydantic v2**                                    | Useful for strict procedure-program schemas, structured VLM outputs, and JSON Schema export. Pydantic’s docs emphasize type-driven validation and JSON Schema generation. ([Pydantic][4])                             |
| Experiment tracking | **MLflow**                                         | Track clustering decisions, VLM calls, optimization metrics, plots, parameters, and exported programs. MLflow supports traces/metrics for agents, LLMs, and ML models. ([MLflow AI Platform][5])                      |
| Geometry            | **NumPy + Shapely + pyproj**                       | NumPy for vectorized fitting; Shapely for planar geometry checks; pyproj/PROJ for coordinate conversion. Shapely is designed for manipulation and analysis of planar geometric objects. ([Shapely][6])                |
| Clustering          | **scikit-learn KMeans first**                      | KMeans is simple, fast, and requires a prespecified cluster count. Its docs note that it can fall into local minima, so multiple restarts are appropriate. ([Scikit-Learn][7])                                        |
| Visualization       | **matplotlib / Plotly**                            | Generate deterministic evidence packs for the VLM and humans.                                                                                                                                                         |
| Storage             | **Parquet + JSON + GeoJSON**                       | Tracks in Parquet; procedure programs in JSON; reconstructable paths in GeoJSON.                                                                                                                                      |

I would **not** start with a managed autonomous-agent platform, multi-agent coding framework, vector database, RAG over AIP documents, or learned neural embedding. Those can come later. The first implementation should be auditable and mostly deterministic.

---

# 3. Design principle: bounded agent, deterministic tools

VLM-PPE should behave like this:

$$
\text{VLM proposes or critiques}
\quad\rightarrow\quad
\text{tool computes}
\quad\rightarrow\quad
\text{schema validates}
\quad\rightarrow\quad
\text{VLM reviews visual evidence}
\quad\rightarrow\quad
\text{deterministic validator accepts/rejects}
$$

The VLM may decide:

$
K=3 \text{ looks better than } K=2,
$

or:

$
\text{cluster 1 looks like a dogleg window}.
$

But the VLM must **not** directly write the geometry. Geometry must come from tools.

This avoids the failure mode you already identified: direct VLM chart parsing is too unreliable. The VLM is still useful because it can inspect overlays, residual-energy charts, and reconstruction plots in a flexible way without requiring a hand-built feature rule for every unusual TMA.

---

# 4. Target output: the procedure program

The engineering team should build toward one central artifact:

```json
{
  "program_id": "VVTS_RWY25L_ARR_CLUSTER_02_DOGLEG_V001",
  "coordinate_system": {
    "type": "local_tangent_plane",
    "origin_lat": 10.8188,
    "origin_lon": 106.6520,
    "unit": "NM"
  },
  "context": {
    "airport": "VVTS",
    "runway": "25L",
    "operation": "arrival",
    "source": "ADS-B"
  },
  "cluster": {
    "cluster_id": 2,
    "track_ids": ["trk001", "trk017", "trk088"],
    "n_tracks": 37
  },
  "template": {
    "type": "polyline",
    "points": [[0.0, 0.0], [4.2, -0.3], [8.9, -0.7], [14.0, -1.1]]
  },
  "interventions": [
    {
      "window_id": "W1",
      "class": "dogleg",
      "s_in_nm": 4.8,
      "s_out_nm": 11.6,
      "tokens": [
        {
          "kind": "LEG",
          "delta_heading_deg": 32.0,
          "length_nm": 5.1
        },
        {
          "kind": "CLOSE",
          "target": "template_s_out"
        }
      ],
      "vertices": [
        [4.7, -0.4],
        [8.9, 2.3],
        [11.4, -0.9]
      ],
      "added_length_nm": 1.42,
      "fit_metrics": {
        "rmse_nm": 0.18,
        "max_error_nm": 0.44,
        "explained_residual_ratio": 0.81
      }
    }
  ],
  "validation": {
    "status": "accepted",
    "deterministic_checks_passed": true,
    "vlm_review": {
      "class_confidence": 0.86,
      "notes": "Single excursion away from common flow, one clear bend, closure back to template."
    }
  },
  "provenance": {
    "pipeline_version": "0.1.0",
    "config_hash": "sha256:...",
    "created_at": "2026-06-11T00:00:00Z"
  }
}
```

This artifact is the “contract” between research and implementation. The client should be able to reconstruct the path from:

[
T(s),
\quad
s_{\text{in}},
\quad
s_{\text{out}},
\quad
(\Delta\psi_j,L_j),
\quad
\text{closure rule}.
]

That is more important than any particular agent framework.

---

# 5. Full VLM-PPE pipeline

## Stage 0 — Dataset ingestion and normalization

Input (to be revised upon inspection of the ADS-B data)

```text
track_id, timestamp, lat, lon, altitude?, groundspeed?, callsign?, runway?, arrival_context?
```

For the first version, use only 2D path geometry:

$
(\text{lat},\text{lon}) \rightarrow (x,y)
$

in nautical miles. Use a local projection centered near the airport or study area. Store the CRS metadata because coordinate mistakes are otherwise hard to debug.

Output:

```text
data/processed/tracks.parquet
data/processed/track_index.parquet
```

Each track should be stored as an ordered polyline:

$
X_i=(x_{i1},x_{i2},\dots,x_{in_i}).
$

Do **not** rely on timestamps for the first version except for ordering. This matches your simplification: paths, not trajectories.

---

## Stage 1 — Arc-length resampling

For clustering and residual charts, resample each path to a fixed number of arc-length points:

$
\tilde X_i(r), \qquad r=0,\dots,R-1.
$

Recommended first setting:

$
R=100.
$

Store both:

1. the raw polyline, for fitting;
2. the resampled polyline, for clustering and visualization.

Output:

```text
data/processed/resampled_tracks.parquet
```

---

## Stage 2 — Shape feature construction

For each resampled track, construct a simple shape vector:

$
f_i=
[x_0,y_0,x_1,y_1,\dots,x_{R-1},y_{R-1},
L_i,
D^{\max}_i].
$

Where:

$
L_i = \text{total path length},
$

$
D^{\max}_i = \text{maximum deviation from a preliminary medoid or baseline}.
$

For v1, keep this simple:

```python
feature = flatten(resampled_xy)
feature = standardize(feature)
```

Optionally add:

```python
total_length_nm
endpoint_heading_deg
max_lateral_spread_nm
```

Do not add too many handcrafted features initially. The goal is to let the cluster panels and VLM review compensate for imperfect feature engineering.

---

## Stage 3 — Candidate clustering

Run KMeans for candidate values:

$
K=1,2,\dots,K_{\max}.
$

Recommended:

$
K_{\max}=8
$

for a single runway/arrival context.

For each $K$:

1. run KMeans with multiple seeds;
2. compute inertia;
3. compute silhouette if there are enough tracks;
4. render cluster overlays;
5. render a “small multiples” plot of all clusters.

KMeans is a good initial tool because it is fast and simple, but because it can fall into local minima, do not rely on a single initialization. ([Scikit-Learn][7])

Output:

```text
artifacts/clustering/k_01/cluster_panel.png
artifacts/clustering/k_02/cluster_panel.png
...
artifacts/clustering/k_metrics.csv
```

---

## Stage 4 — VLM cluster review

The VLM receives:

1. one panel per (K);
2. inertia/silhouette chart;
3. cluster overlays;
4. examples of tracks near each cluster medoid;
5. instruction: “Use ADS-B plots only. Do not infer from AIP.”

The VLM returns structured JSON:

```json
{
  "chosen_k": 3,
  "confidence": 0.78,
  "rationale": [
    "K=2 merges two visually distinct downwind-extension groups.",
    "K=4 splits one coherent group without a meaningful new procedure."
  ],
  "clusters_to_recheck": [1],
  "suggested_action": "accept"
}
```

The agent is allowed to call the clustering tool again only through a bounded policy:

```text
maximum clustering retries = 2
maximum Kmax expansion = 12
```

No infinite loops.

---

# 6. Practical-template extraction inside each cluster

For each accepted cluster, construct a practical template.

## ADS-B-only data 

I recommend:

1. compute the cluster medoid;
2. identify common prefix and suffix;
3. use the shortest low-residual track as the initial baseline;
4. let intervention windows absorb high-variance portions.

The medoid is safer than the mean for v1:

$$
i^\star
=

\arg\min_i
\sum_j
d_{\text{aligned}}(\tilde X_i,\tilde X_j).
$$

Then:

$$
T_{\text{cluster}}=\tilde X_{i^\star}.
$$

This gives the agent a concrete reference path without inventing an averaged path that no aircraft actually flew.

---

# 7. Residual-energy chart and intervention-window detection

For a cluster with tracks $\tilde X_i(r)$ and template $T(r)$, compute residuals:

$$e_i(r)=|\tilde X_i(r)-T(r)|.$$

Then compute robust residual energy:

$$E(r)=\operatorname{median}_i \left(e_i(r)^2\right).$$

Also compute heading dispersion:

$$V_\psi(r) = 1 - \left| \frac{1}{N} \sum_i \exp(j\psi_i(r)) \right|.$$

A candidate intervention window is a contiguous interval where either:

$$E(r) > \operatorname{median}(E)+\lambda\operatorname{MAD}(E),$$

or:

$$V_\psi(r) > \tau_\psi.$$

Recommended initial values:

```yaml
residual_energy_lambda: 3.0
min_window_length_nm: 2.0
merge_windows_gap_nm: 1.0
heading_dispersion_threshold: 0.25
```

Treat these as engineering defaults, not learned hyperparameters.

For each cluster, render:

```text
1. track overlay
2. practical template / medoid
3. residual-energy curve E(r)
4. heading-dispersion curve
5. highlighted candidate windows
6. sampled tracks from the window
```

The VLM then reviews this evidence and returns:

```json
{
  "cluster_id": 2,
  "windows": [
    {
      "window_id": "W1",
      "start_r": 32,
      "end_r": 61,
      "class_hint": "dogleg",
      "confidence": 0.82,
      "visual_reason": "One clear outward vector and one closure leg back to common flow."
    }
  ],
  "outlier_notes": [
    "Two tracks appear to make a direct shortcut after the merge region."
  ]
}
```

---

# 8. Intervention classes for v1

Keep the class set small:

[
\mathcal C =
{
\text{no-stretch},
\text{dogleg},
\text{trombone},
\text{PMS},
\text{other}
}.
]

Do not include holding as a normal class in v1. Put it in “other” and study it in Phase 2.

## 8.1 No-stretch

No deviation program:

[
P=T.
]

Accept if:

```text
max cross-track deviation small
added path length small
no residual-energy peak
```

## 8.2 Dogleg

One free midpoint:

[
A\rightarrow G_1\rightarrow B.
]

Program:

[
D=
(s_{\text{in}},s_{\text{out}},\Delta\psi_1,L_1).
]

The closure leg is:

[
G_1\rightarrow T(s_{\text{out}}).
]

## 8.3 Trombone

Two free midpoints:

[
A\rightarrow G_1\rightarrow G_2\rightarrow B.
]

Program:

[
D=
(s_{\text{in}},s_{\text{out}},
\Delta\psi_1,L_1,
\Delta\psi_2,L_2).
]

## 8.4 PMS

Treat PMS separately. EUROCONTROL describes Point Merge as a systemised arrival-sequencing method based on a merge point and predefined equidistant sequencing legs; sequencing is achieved by a “direct-to” instruction to the merge point, and the legs are used for delay/path stretching. ([EUROCONTROL][8])

So PMS should not be forced into a generic dogleg model.

A PMS program should be:

[
D_{\text{PMS}}=(\text{sequencing leg},M,s_{\text{release}}),
]

where (M) is the merge point and (s_{\text{release}}) is the position along the sequencing leg at which the direct-to instruction appears to have occurred.

## 8.5 Other

Anything complex, looping, self-crossing, ambiguous, or rare becomes:

```json
"class": "other"
```

This protects the v1 model from overfitting.

---

# 9. Optimization spine

This is the mathematical heart of the implementation.

For each track and intervention window, define:

[
Q=(q_0,q_1,\dots,q_N)
]

as the observed excursion inside the window.

Let:

[
A=T(s_{\text{in}}),
\qquad
B=T(s_{\text{out}}).
]

For class dogleg:

[
A\rightarrow G_1\rightarrow B.
]

For class trombone:

[
A\rightarrow G_1\rightarrow G_2\rightarrow B.
]

For arbitrary fixed (M):

[
P_0=A,\quad P_1=G_1,\quad \dots,\quad P_M=G_M,\quad P_{M+1}=B.
]

The objective is:

[
J=
\sum_{\ell=0}^{M}
\sum_{i=b_\ell}^{b_{\ell+1}}
w_i
d^2(q_i,[P_\ell,P_{\ell+1}]).
]

Where:

[
0=b_0<b_1<\dots<b_M<b_{M+1}=N.
]

The method should be:

[
\boxed{
\text{breakpoint scan / dynamic programming}
\rightarrow
\text{continuous refinement}
\rightarrow
\text{feasibility validation}
}
]

---

## 9.1 Dogleg fitting

Scan every possible breakpoint (b):

[
G_1=q_b.
]

Compute:

[
J(b)
====

\operatorname{SegErr}(A,q_b;0,b)
+
\operatorname{SegErr}(q_b,B;b,N).
]

Choose:

[
b^\star=\arg\min_b J(b).
]

Then initialize:

[
G_1^{(0)}=q_{b^\star}.
]

After this, refine (G_1) continuously.

Simplest refinement:

1. fit a line through (A) to the first group;
2. fit a line through (B) to the second group;
3. intersect the two fitted lines;
4. use the intersection as (G_1);
5. fall back to (q_{b^\star}) if the lines are nearly parallel.

Then derive:

[
L_1=|G_1-A|,
]

[
\Delta\psi_1=
\operatorname{wrap}
\left(
\operatorname{heading}(A,G_1)
-----------------------------

\psi_T(s_{\text{in}})
\right).
]

The closure length is:

[
L_{\text{close}}=|B-G_1|.
]

The added path length is:

[
\Delta L=L_1+L_{\text{close}}-(s_{\text{out}}-s_{\text{in}}).
]

---

## 9.2 Trombone fitting

Scan ordered pairs:

[
b_1<b_2.
]

Set:

[
G_1=q_{b_1},
\qquad
G_2=q_{b_2}.
]

Compute:

[
J(b_1,b_2)
==========

\operatorname{SegErr}(A,q_{b_1};0,b_1)
+
\operatorname{SegErr}(q_{b_1},q_{b_2};b_1,b_2)
+
\operatorname{SegErr}(q_{b_2},B;b_2,N).
]

Choose the best pair.

Then refine continuously by fitting three line trends:

[
\ell_0: A\rightarrow G_1,
]

[
\ell_1: G_1\rightarrow G_2,
]

[
\ell_2: G_2\rightarrow B.
]

Then:

[
G_1=\ell_0\cap\ell_1,
]

[
G_2=\ell_1\cap\ell_2.
]

If intersections are unstable, keep the breakpoint vertices.

---

## 9.3 Fixed-(M) dynamic programming

For (M>2), use dynamic programming, not gradient descent.

Precompute segment costs:

[
C(i,j)=\operatorname{SegErr}(q_i,q_j;i,j).
]

Then:

[
D(1,j)=\operatorname{SegErr}(A,q_j;0,j).
]

For (m=2,\dots,M):

[
D(m,j)=\min_{i<j}\left[D(m-1,i)+C(i,j)\right].
]

Finalize with:

[
J^\star_M=\min_j\left[D(M,j)+\operatorname{SegErr}(q_j,B;j,N)\right].
]

This gives the best ordered observed-point vertices. Continuous refinement can follow, but v1 should cap:

[
M_{\max}=2.
]

Everything requiring (M>2) should usually become “other.”

---

# 10. Program fitting policy

For each track-window, fit candidate models:

[
M=0,
\quad
M=1,
\quad
M=2.
]

Compute:

```text
RMSE_0
RMSE_1
RMSE_2
max_error_0
max_error_1
max_error_2
added_length_nm
leg_lengths
heading_changes
```

Then apply deterministic class rules:

```python
if rmse_0 < eps_no_stretch and added_length < min_added_length:
    class = "no_stretch"

elif class_hint == "PMS" and pms_geometry_check_passes:
    class = "PMS"

elif rmse_1 < eps_dogleg and feasibility_passes(M=1):
    class = "dogleg"

elif rmse_2 < eps_trombone and rmse_2 < improvement_ratio * rmse_1 and feasibility_passes(M=2):
    class = "trombone"

else:
    class = "other"
```

Recommended initial config:

```yaml
eps_no_stretch_rmse_nm: 0.15
eps_dogleg_rmse_nm: 0.35
eps_trombone_rmse_nm: 0.35
max_allowed_error_nm: 0.8
improvement_ratio_for_extra_vertex: 0.70
min_leg_length_nm: 1.5
min_heading_change_deg: 12.0
max_added_length_nm: 25.0
allow_self_crossing: false
```

These should be easy-to-read operational thresholds, not Bayesian priors.

---

# 11. Cluster-level aggregation

Do not fit only the cluster medoid. Fit every track in the cluster, then aggregate.

For cluster (c), collect:

[
\Theta_c=
{\theta_{c1},\theta_{c2},\dots,\theta_{cN_c}}.
]

For doglegs:

[
\theta_i=(s_{\text{in}},s_{\text{out}},\Delta\psi_1,L_1,\Delta L).
]

For trombones:

[
\theta_i=(s_{\text{in}},s_{\text{out}},\Delta\psi_1,L_1,\Delta\psi_2,L_2,\Delta L).
]

Export:

```json
{
  "class": "dogleg",
  "n_tracks": 37,
  "parameter_summary": {
    "s_in_nm": {"median": 4.8, "p10": 4.2, "p90": 5.4},
    "s_out_nm": {"median": 11.6, "p10": 10.8, "p90": 12.7},
    "delta_heading_deg": {"median": 32.0, "p10": 24.0, "p90": 39.0},
    "leg_length_nm": {"median": 5.1, "p10": 3.8, "p90": 7.2},
    "added_length_nm": {"median": 1.42, "p10": 0.7, "p90": 2.9}
  }
}
```

For generation or reconstruction examples, use either:

1. median program;
2. medoid program;
3. empirical resampling of observed programs.

Avoid fitted multivariate distributions in v1.

---

# 12. Agent graph

The LangGraph state can be:

```python
class PPEState(BaseModel):
    dataset_id: str
    context: dict
    tracks_path: str
    resampled_tracks_path: str | None = None

    candidate_k_values: list[int] = []
    clustering_runs: list[dict] = []
    chosen_k: int | None = None
    cluster_assignments_path: str | None = None

    cluster_diagnostics: list[dict] = []
    intervention_windows: list[dict] = []
    fitted_programs: list[dict] = []

    pending_actions: list[dict] = []
    vlm_reviews: list[dict] = []
    errors: list[dict] = []
```

Recommended graph:

```text
START
  ↓
ingest_tracks
  ↓
resample_tracks
  ↓
build_shape_features
  ↓
run_candidate_clustering
  ↓
render_cluster_diagnostics
  ↓
vlm_select_k
  ↓
validate_cluster_choice
  ├── retry clustering, if allowed
  └── accept
        ↓
extract_cluster_templates
        ↓
compute_residual_windows
        ↓
render_window_diagnostics
        ↓
vlm_classify_windows
        ↓
fit_programs
        ↓
render_reconstruction_diagnostics
        ↓
vlm_review_reconstructions
        ↓
deterministic_acceptance_gate
        ├── split/retry/other
        └── export_programs
              ↓
END
```

The most important guardrail is this:

```text
VLM can request tool actions.
VLM cannot directly create final procedure geometry.
```

---

# 13. Suggested repository spine

```text
vlm-ppe/
  pyproject.toml
  README.md
  configs/
    default.yaml
    vvts_arrivals.yaml

  data/
    raw/
    processed/

  artifacts/
    clustering/
    windows/
    reconstructions/
    reports/

  src/vlm_ppe/
    __init__.py

    schemas.py
    constants.py

    io/
      adsb_loader.py
      parquet_store.py
      geojson_export.py

    geo/
      projection.py
      polyline.py
      distance.py
      resample.py

    clustering/
      features.py
      kmeans_runner.py
      metrics.py
      medoid.py

    diagnostics/
      plots_tracks.py
      plots_clustering.py
      plots_residual_energy.py
      plots_reconstruction.py

    windows/
      residual_energy.py
      heading_dispersion.py
      detect_windows.py

    optimization/
      segment_error.py
      breakpoint_scan.py
      dynamic_programming.py
      continuous_refinement.py
      fit_dogleg.py
      fit_trombone.py
      fit_pms.py

    validation/
      feasibility.py
      reconstruction_metrics.py
      acceptance.py

    agents/
      graph.py
      prompts.py
      tools.py
      vlm_client.py

    reporting/
      markdown_report.py
      summary_tables.py

  tests/
    test_projection.py
    test_resample.py
    test_segment_error.py
    test_breakpoint_scan.py
    test_refinement.py
    test_program_roundtrip.py
    test_synthetic_recovery.py
```

---

# 14. Core schema spine

Use Pydantic models for every VLM output and every exported program.

```python
from typing import Literal
from pydantic import BaseModel, Field


Point2D = tuple[float, float]


class CoordinateSystem(BaseModel):
    type: Literal["local_tangent_plane", "projected_crs"]
    unit: Literal["NM", "m"]
    origin_lat: float | None = None
    origin_lon: float | None = None
    epsg: int | None = None


class Polyline(BaseModel):
    points: list[Point2D]


class LegToken(BaseModel):
    kind: Literal["LEG"]
    delta_heading_deg: float
    length_nm: float


class CloseToken(BaseModel):
    kind: Literal["CLOSE"]
    target: Literal["template_s_out", "merge_point"]


class PMSToken(BaseModel):
    kind: Literal["PMS_DIRECT"]
    merge_point: Point2D
    release_s_nm: float


Token = LegToken | CloseToken | PMSToken


class FitMetrics(BaseModel):
    rmse_nm: float
    max_error_nm: float
    added_length_nm: float
    explained_residual_ratio: float | None = None


class InterventionProgram(BaseModel):
    window_id: str
    class_name: Literal["no_stretch", "dogleg", "trombone", "PMS", "other"]
    s_in_nm: float | None = None
    s_out_nm: float | None = None
    tokens: list[Token]
    vertices: list[Point2D]
    fit_metrics: FitMetrics
    source_track_ids: list[str]


class ProcedureProgram(BaseModel):
    program_id: str
    coordinate_system: CoordinateSystem
    context: dict
    cluster_id: int
    template: Polyline
    interventions: list[InterventionProgram]
    validation: dict
    provenance: dict
```

This is the object the client should consume.

---

# 15. Core optimization API

Give the engineering team these function signatures first:

```python
def point_to_segment_distance_sq(
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> float:
    """Squared distance from point q to segment [u, v]."""


def segment_error(
    q: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    i0: int,
    i1: int,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted squared error from q[i0:i1+1] to segment [u, v]."""


def fit_dogleg_breakpoint_scan(
    q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    config: FitConfig,
) -> DoglegFit:
    """Find G1 by breakpoint scan, then optional continuous refinement."""


def fit_trombone_breakpoint_scan(
    q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    config: FitConfig,
) -> TromboneFit:
    """Find G1,G2 by ordered pair scan, then optional continuous refinement."""


def fit_fixed_m_dynamic_programming(
    q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    m: int,
    config: FitConfig,
) -> SegmentedPolylineFit:
    """General fixed-M segmented polyline fit."""


def reconstruct_program(
    template: Polyline,
    intervention: InterventionProgram,
) -> Polyline:
    """Reconstruct the path from template and tokens."""
```

For v1, implement dogleg and trombone first. Dynamic programming can be implemented immediately after but capped at (M=2).

---

# 16. VLM prompt contracts

## Cluster-selection prompt

```text
You are reviewing ADS-B arrival tracks. Use only the trajectory plots and metrics shown.
Do not infer from AIP charts, waypoint names, or expected procedures.

Task:
Choose the most plausible number of practical path clusters.

Prefer:
- fewer clusters when the difference is only noise;
- more clusters when paths have clearly different common geometry;
- an "outlier" bucket instead of forcing rare tracks into normal clusters.

Return valid JSON matching the schema:
{
  "chosen_k": int,
  "confidence": float,
  "rationale": [string],
  "clusters_to_recheck": [int],
  "suggested_action": "accept" | "retry_with_larger_k" | "retry_with_smaller_k" | "human_review"
}
```

## Window-classification prompt

```text
You are reviewing one trajectory cluster.

Inputs:
- overlay of ADS-B tracks;
- practical template / medoid;
- residual-energy curve;
- heading-dispersion curve;
- candidate intervention windows.

Classify each window as:
no_stretch, dogleg, trombone, PMS, or other.

Definitions:
- dogleg: one outward vector-like leg and one closure/rejoin leg;
- trombone: two vector-like legs before closure/rejoin;
- PMS: sequencing leg followed by direct-to common merge point;
- other: holding, looping, direct shortcut, unclear, or too rare.

Return valid JSON only.
```

## Reconstruction-review prompt

```text
You are reviewing fitted procedure programs against ADS-B tracks.

Check:
- Does the reconstructed polyline follow the observed intervention?
- Does the class label match the visual evidence?
- Are there obvious outliers that should not be included?
- Is the fit over-complicated?

Return:
{
  "status": "accept" | "reject" | "split_cluster" | "mark_other" | "human_review",
  "confidence": float,
  "reasons": [string],
  "track_ids_to_exclude": [string]
}
```

The VLM’s answer should always be parsed by Pydantic. If parsing fails, the graph should retry once with a shorter prompt and then route to human review.

---

# 17. Deterministic validation gates

A fitted program should be accepted only if these pass:

```text
1. s_out > s_in
2. every leg length >= min_leg_length_nm
3. heading changes are meaningful for inserted vertices
4. closure leg length is not absurd relative to window length
5. added_length_nm >= -small_tolerance
6. reconstruction RMSE <= class threshold
7. max reconstruction error <= class threshold
8. no self-crossing unless class == other or holding
9. PMS only accepted if common merge-point geometry is present
10. output reconstructs exactly from the exported JSON
```

This matters because the VLM may visually accept a cluster that the geometry cannot support. The deterministic gate should win.

---

# 18. Residual/reconstruction diagnostics to generate

For each cluster:

```text
cluster_overlay.png
cluster_medoids.png
k_selection_metrics.png
residual_energy.png
heading_dispersion.png
window_highlights.png
```

For each fitted class:

```text
fit_overlay_all_tracks.png
fit_overlay_median_track.png
observed_vs_reconstructed_examples.png
parameter_histograms.png
added_length_distribution.png
error_distribution.png
outlier_examples.png
```

The VLM should inspect images, but the final report should include the numeric metrics too.

---

# 19. Evaluation metrics

Use operationally meaningful metrics.

## Track-level

[
\operatorname{RMSE}
===================

\sqrt{
\frac{1}{N}
\sum_i
d^2(q_i,\hat P)
}.
]

Also compute:

```text
max point-to-polyline error
added path length error
rejoin error
divergence/rejoin location error, if manually labelled
```

## Cluster-level

```text
median RMSE
90th percentile RMSE
outlier rate
number of accepted tracks
number of rejected tracks
class distribution
```

## Procedure-level

```text
distribution of s_in
distribution of s_out
distribution of delta_heading
distribution of leg length
distribution of added path length
maximum lateral envelope
```

## Agent-level

```text
number of VLM retries
number of invalid JSON responses
number of human-review routes
number of accepted/rejected clusters
agreement with human labels on sampled cases
```

---

# 20. Phase 2 outlier study

Phase 2 should not complicate the v1 normal-procedure extractor.

For outliers, add two detectors.

## 20.1 Direct-link detector

Look for long straight segments that skip part of the practical template.

For each outlier track, test candidate direct links:

[
T(s_a)\rightarrow T(s_b),
\qquad s_b>s_a.
]

If a large observed segment lies close to this shortcut, export:

```json
{
  "outlier_type": "possible_direct",
  "from_s_nm": 5.4,
  "to_s_nm": 13.2,
  "supporting_track_ids": ["trk011", "trk045"],
  "notes": "Observed repeatedly in low-traffic samples."
}
```

Do not infer the reason yet. Only report that the direct link was historically observed.

## 20.2 Holding / loop detector

Flag tracks with:

```text
self-crossing
large cumulative heading change
repeated loop-like geometry
high path length / endpoint distance ratio
```

Export:

```json
{
  "outlier_type": "possible_holding",
  "supporting_track_ids": ["trk009", "trk052"],
  "loop_count_estimate": 1,
  "status": "phase_2_manual_review"
}
```

---

# 21. Implementation milestones

## Milestone A — Deterministic geometry core

Deliver:

```text
resampling
point-to-segment distance
dogleg fitting
trombone fitting
program reconstruction
synthetic tests
```

No VLM yet.

Success condition:

```text
Synthetic doglegs and trombones recover G1/G2 and added length within tolerance.
```

## Milestone B — Clustering and residual windows

Deliver:

```text
shape features
KMeans candidate runs
cluster plots
residual-energy windows
heading-dispersion plots
```

Success condition:

```text
A human can inspect generated plots and identify plausible clusters/windows.
```

## Milestone C — VLM review loop

Deliver:

```text
LangGraph state machine
VLM cluster-selection node
VLM window-classification node
Pydantic structured-output validation
retry/human-review routing
```

Success condition:

```text
The VLM can select K and classify windows, but all geometry still comes from deterministic tools.
```

## Milestone D — Program export

Deliver:

```text
ProcedureProgram JSON
GeoJSON reconstruction
Markdown report
MLflow run logs
```

Success condition:

```text
A client can reconstruct every accepted procedure from exported JSON alone.
```

## Milestone E — Outlier phase

Deliver:

```text
possible direct-link detector
possible holding detector
outlier report
```

Success condition:

```text
Rare cases are described without contaminating dogleg/trombone/PMS extraction.
```

---

# 22. Testing plan

The tests should be explicit and geometry-heavy.

## Unit tests

```text
test point-to-segment distance
test arc-length resampling preserves endpoints
test polyline length
test projection to template
test heading wrapping
test added length calculation
```

## Synthetic recovery tests

Generate known doglegs:

[
A\rightarrow G_1\rightarrow B
]

with noise. Verify that the fitted (G_1), (\Delta\psi_1), (L_1), and (\Delta L) are recovered.

Generate known trombones:

[
A\rightarrow G_1\rightarrow G_2\rightarrow B.
]

Verify ordered breakpoint recovery.

## Invariance tests

The fitted program should be invariant to:

```text
translation
rotation
minor resampling changes
small observational noise
```

## Round-trip tests

For every exported program:

```text
JSON program -> reconstructed path -> metrics recomputed
```

The recomputed metrics should match the stored metrics.

---

# 23. Key failure modes and guardrails

| Failure mode                    | Guardrail                                                                 |
| ------------------------------- | ------------------------------------------------------------------------- |
| VLM hallucinates a procedure    | VLM cannot write geometry; only deterministic tools export programs       |
| KMeans splits noise             | VLM sees cluster panels; deterministic metrics and human review available |
| KMeans misses rare class        | Keep outlier bucket; do not force all tracks into normal clusters         |
| Residual window too broad       | Hysteresis, minimum/maximum length, heading-dispersion confirmation       |
| Dogleg/trombone overfit         | Fixed token count, minimum leg length, improvement-ratio rule             |
| PMS confused with dogleg        | PMS accepted only when merge-point/direct-to structure is detected        |
| AIP chart parsing fails         | AIP is optional anchor/naming source; ADS-B tracks remain primary         |
| Coordinate mistakes             | Store CRS/origin/unit in every artifact                                   |
| VLM output malformed JSON       | Pydantic parse gate and retry/human-review route                          |
| Rare holding contaminates model | Label “other” in v1; study in Phase 2                                     |

---

# 24. What I would not implement in v1

I would explicitly defer:

```text
full HMM / HSMM
Bayesian priors
neural trajectory embeddings
CNN classification
transformers
direct VLM chart parsing as source of truth
automatic causal explanation for why outliers happened
3D / energy modeling
turn-radius modeling
real-time separation assurance
```

This is not because those ideas are bad. It is because the v1 contribution is cleaner:

[
\boxed{
\text{agent-guided extraction of simple, reconstructable, practical procedure programs from ADS-B}
}
]

---

# 25. The “implementation spine” in one page

The team can start from this:

```python
def run_vlm_ppe(dataset_id: str, config: PPEConfig) -> list[ProcedureProgram]:
    tracks = load_adsb_tracks(config.input_path)
    tracks_xy = project_tracks_to_local_xy(tracks, config.projection)
    tracks_clean = clean_and_filter_tracks(tracks_xy, config.qc)

    resampled = arc_length_resample_tracks(tracks_clean, n_points=config.n_resample)
    features = build_shape_features(resampled, config.features)

    clustering_runs = []
    for k in range(1, config.k_max + 1):
        labels = run_kmeans(features, k=k, n_init=config.kmeans_n_init)
        metrics = compute_cluster_metrics(features, labels)
        plots = render_cluster_diagnostics(resampled, labels, metrics)
        clustering_runs.append({"k": k, "labels": labels, "metrics": metrics, "plots": plots})

    chosen = vlm_select_cluster_count(clustering_runs)
    labels = clustering_runs[chosen.k]["labels"]

    programs = []

    for cluster_id in unique(labels):
        cluster_tracks = select_tracks(tracks_clean, labels, cluster_id)
        template = extract_practical_template(cluster_tracks, config.template)

        residual = compute_residual_energy(cluster_tracks, template)
        heading_var = compute_heading_dispersion(cluster_tracks, template)
        windows = detect_intervention_windows(residual, heading_var, config.windows)

        window_review = vlm_classify_windows(
            cluster_tracks=cluster_tracks,
            template=template,
            windows=windows,
            plots=render_window_diagnostics(cluster_tracks, template, residual, heading_var, windows),
        )

        for window in window_review.windows:
            candidate_fits = []

            if window.class_hint in ["dogleg", "other"]:
                candidate_fits.append(fit_dogleg_cluster(cluster_tracks, template, window, config.fit))

            if window.class_hint in ["trombone", "other"]:
                candidate_fits.append(fit_trombone_cluster(cluster_tracks, template, window, config.fit))

            if window.class_hint in ["PMS", "other"]:
                candidate_fits.append(fit_pms_cluster(cluster_tracks, template, window, config.fit))

            accepted = choose_and_validate_fit(candidate_fits, config.validation)

            reconstruction_plots = render_reconstruction_diagnostics(
                cluster_tracks, template, accepted
            )

            vlm_review = vlm_review_reconstruction(reconstruction_plots, accepted)

            final_program = deterministic_acceptance_gate(
                accepted,
                vlm_review,
                config.validation,
            )

            programs.append(final_program)

    export_programs_json(programs, config.output_dir)
    export_programs_geojson(programs, config.output_dir)
    write_markdown_report(programs, config.output_dir)
    log_to_mlflow(dataset_id, config, programs)

    return programs
```

---

# 26. Final recommendation

Build VLM-PPE as a **tool-using visual reasoning loop around a deterministic procedure-program extractor**.

The best first version is:

[
\boxed{
\text{ADS-B tracks}
\rightarrow
\text{shape clusters}
\rightarrow
\text{residual-energy windows}
\rightarrow
\text{VLM-reviewed class hints}
\rightarrow
\text{fixed-token polyline optimization}
\rightarrow
\text{schema-validated practical procedure programs}
}
]

This preserves the flexibility of an agentic loop while keeping the actual geometry simple, reconstructable, and defensible. The research novelty should be framed as a practical extraction system whose output is a **procedure-anchored, closure-constrained controller-instruction representation**, not just a trajectory cluster or a black-box generative model. The attached literature note already identifies this as the useful wedge: existing work tends to generate trajectories without controller-level structure or recognize maneuver classes without producing reconstructable, delay-aware procedure programs. 

[1]: https://www.icao.int/airnavigation/aeronautical-information-management "Aeronautical Information Management"
[2]: https://docs.langchain.com/oss/python/langgraph/overview "LangGraph overview - Docs by LangChain"
[3]: https://ai.google.dev/gemini-api/docs/models "Models  |  Gemini API  |  Google AI for Developers"
[4]: https://pydantic.dev/docs/validation/latest/get-started/ "Welcome to Pydantic | Pydantic Docs"
[5]: https://mlflow.org/ "MLflow - Open Source AI Platform for Agents, LLMs & Models"
[6]: https://shapely.readthedocs.io/ "Shapely — Shapely 2.1.2 documentation"
[7]: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html "KMeans — scikit-learn 1.9.0 documentation"
[8]: https://www.eurocontrol.int/point-merge "Point Merge | EUROCONTROL"
