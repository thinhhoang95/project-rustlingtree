# Downloading Airport Charts Automatically
> Note that we will use CIFP for procedure extraction, but these charts will provide visual confirmation.

1. Download the airport's data from FAA's ADIP: `https://adip.faa.gov/agis/public/#/airportCharts/DFW` as a single `.mhtml` file.
2. Run the script `node src/scenario/adip_resources/extract-kdfw-charts.js data/adip/kdfw_adip_resources.mhtml --out data/adip/kdfw_adip_resources.json` to extract the chart URLs to a single JSON file.
3. Use the script `src/scenario/adip_resources/download_kdfw_charts.py`. Despite its name, you can use any JSON manifest you just created for any airport, and it will automatically download all associated PDF resources to a local directory: `python -m scenario.adip_resources.download_kdfw_charts data/adip/kdfw_adip_resources.json --output-dir data/adip/charts`.
4. Convert all PDF files to JPG files so that the LLM could read them with `python -m scenario.adip_resources.convert_adip_charts_to_jpg data/adip/charts --output-dir data/adip/charts/img`. --> this step is unreliable, you should not do it, use the CIFP procedure below.

# Extraction of CIFP Resources
1. Download CIFP data from FAA. Put it in data/cifp, rename to FAACIFP18.txt (add .txt extension).
2. Run the notebook `src/scenario/cifp_parser/parse.ipynb`, it will produce `data/kdfw_procs/` procedure CSV files.

If you want to know what the fields mean, check out `docs/cifp/arinc424_route_and_section_code_reference.md`.

# ADS-B Demand Data Download
The script to realize this is `src/scenario/demand_opensky/1_1download_ostrino.py`. Make sure you have ostrino CLI ready in the project root. This will give `data/adsb/raw` CSV files. You need to provide the datetime range of data to be downloaded, as well as the timezone.

Then you can run the script `src/scenario/demand_opensky/extract_departures_and_arrivals.py` to automatically extract takeoffs and landings. Note that it will also download an authoritative departures and arrivals catalog from OpenSky to cross-validate the data validity. You must supply the same datetime range and timezone as in the ostrino download step.

### ADS-B Data Processing
Generate the arrival, departure catalogs, and compress data for the rustlingleaves client.

### Arrival and Departure Catalog
```bash 
python src/scenario/demand_opensky/extract_departures_and_arrivals.py \
  --from-datetime "2025-04-01T00:00:00" \
  --to-datetime "2025-04-01T23:59:59" \
  --timezone "America/Chicago" \
  --split-gap-seconds 1500
```

This step writes the derived landings/departures catalog, the fix-sequence catalog, and the authoritative departures/arrivals catalog into `data/adsb/catalogs`. You might want to check that the derived departure/arrival catalog should not miss too many flights (possibly less than 1-2%) of the OpenSky's authoritative catalog in the console output.

### Compress the Data
python src/scenario/trajectory_compressor/cli.py --landings-departures-catalog

This writes the full ADS-B compressed trajectory file used for departures at `data/adsb/compressed/adsb_compressed_flights.jsonl`. SIMAP arrival artifacts are written separately by `scenario-manager-precompute-artifact` to `data/artifacts/simap_arrival_flights.jsonl`.

# Hail Mary

After finishing the catalog generation above, just go straight to the `build_offline_corpus` step below. In the following sections, we will only detail the algorithms, but the bash command needed to be run is only related to build_offline_corpus. 

## 1. Runway Attribution 
File: `/Volumes/CrucialX/project-rustlingtree/src/scenario/demand_opensky/adsb_catalog_processing.py`

Example: `ENY3938M1a379dd`, approaching southbound near KDFW.

1. Find eligible points  
   Keep ADS-B samples within 5 km of a threshold and below 10,000 ft.

2. Apply the direction gate  
   Its final heading is about `181°`.


The heading error is the smallest circular difference:

```text
error = abs(((aircraft_heading - runway_heading + 180) % 360) - 180)
```

It ranges from `0°` to `180°`:

- `0°`: perfectly aligned
- `45°`: passes
- `90°`: passes at the boundary
- `120°`: rejected
- `180°`: reciprocal direction, rejected

3. Define one encounter  
   Starting from the first direction-compatible sample, inspect the next 300 seconds.

4. Find the closest point for each candidate  
   At the best arrival sample:

5. Choose by distance  
   `18R` wins. Approach strength and heading error are only tie-breakers.

6. Record the event  
   The flight is stored as an arrival on `18R`, using that 8.7 m sample as its arrival event.


For the final southbound sample:

| Runway | Runway heading | Error | Distance |
|---|---:|---:|---:|
| `18R` | 180.245° | 0.734° | 8.7 m |
| `18L` | 180.248° | 0.732° | 363 m |
| `17R` | 180.257° | 0.722° | 2,309 m |
| `17C` | 180.257° | 0.722° | 2,674 m |
| `13R` | 139.188° | 41.791° | 2,748 m |

All pass the heading gate. `18R` wins because distance is the primary ranking criterion.

Some reciprocal ends also produce candidates from an earlier part of the track. For example, an earlier `305.6°` vector is only `54.7°` from runway heading `0°`, so it passes the gate for several northbound ends. Those candidates remain over 1.6 km away and therefore lose to `18R`.

So the distinction is:

```text
±90° heading gate → eliminates clearly incompatible motion
closest distance   → actually selects the runway
```

The tolerance is intentionally a reciprocal-direction veto, not a precise final-alignment test. Tightening it to roughly `30–45°` would exclude more vectoring candidates, but could reject legitimate turning approaches with sparse ADS-B sampling.



## 2. Clustering
The data is first partitioned by runway, then individual flight trajectories are clustered independently inside each runway partition.
The flow is:

1. The multi-runway builder enumerates runway IDs and processes each runway separately in [pipeline.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/clustering/pipeline.py:914).
2. `_partition_arrivals()` filters catalogued flights by airport and runway in [pipeline.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/clustering/pipeline.py:572).
3. Each accepted flight track is aligned to that runway threshold and resampled in [pipeline.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/clustering/pipeline.py:601).
4. The mapping of individual `flight_id → resampled trajectory` is passed into `build_cluster_library()` in [pipeline.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/clustering/pipeline.py:891).
5. Every trajectory becomes one flattened, standardized feature row in [features.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/clustering/features.py:141).
6. HDBSCAN clusters those flight-level rows in [artifact.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/clustering/artifact.py:290) and [hdbscan_runner.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/clustering/hdbscan_runner.py:359).
7. Cluster medoids are calculated only after HDBSCAN has produced its labels in [artifact.py](/Volumes/CrucialX/project-rustlingtree/src/hailmary/clustering/artifact.py:305).

So conceptually:

```text
All arrivals
  ├── RW18R flights ── HDBSCAN ── clusters 0, 1, 2...
  ├── RW17C flights ── HDBSCAN ── clusters 0, 1...
  └── RW36L flights ── HDBSCAN ── clusters 0, 1, 2...
```

There is no subsequent HDBSCAN across the resulting runway clusters or their medoids. The final cluster identity is therefore qualified as `(airport, runway, cluster_id)`, such as `KDFW:RW18R:2`.

## 2.1 HDBSCAN Details

#### 1. Trajectory preparation

For each runway:

1. Retain the final inbound trajectory from the 50 NM terminal boundary to landing.
2. Project latitude/longitude into runway-centered local east/north coordinates.
3. Align every trajectory to the runway threshold.
4. Resample each path to 128 equally spaced geometric stations.
5. Flatten the station coordinates and standardize every feature across the runway cohort.

#### 2. Global HDBSCAN sweep

Every runway evaluates the same 40 candidates:

- `min_cluster_size ∈ {4, 8, 12, 16, 24}`
- `min_samples ∈ {None, 3, 5, 8}`
- selection method ∈ `{EOM, leaf}`

#### 3. Candidate diagnostics

Each candidate is evaluated using:

- **Silhouette:** separation between clusters, excluding noise.
- **Mean persistence:** density stability of HDBSCAN clusters.
- **Coverage:** fraction of trajectories receiving a non-noise label.
- **Fragmentation:** cluster count divided by clustered trajectory count.
- **Physical dispersion:** 90th-percentile physical distance from each cluster’s mean trajectory, normalized by terminal-entry radius. The worst cluster is used.
- **Entry-bearing span:** shortest circular arc containing 90% of a cluster’s terminal-entry bearings.

Per-cluster membership, dispersion, and entry-bearing diagnostics are persisted and printed during corpus construction.

#### 4. Candidate rejection

A candidate is rejected if it has:

- fewer than 2 clusters;
- more than 60% noise;
- fragmentation above 0.25; or
- a cluster whose 90% entry-bearing span exceeds 60°.

The bearing constraint prevents arrivals from materially different approach directions from being merged.

#### 5. Candidate scoring

Accepted candidates receive:

```text
score =
  0.35 × silhouette
+ 0.15 × mean persistence
+ 0.35 × coverage
- 0.15 × fragmentation
- 0.10 × worst physical dispersion
```

Higher scores are better. Exact ties are resolved deterministically using serialized parameter order.

The weighting favors well-separated clusters that cover repeatable approach patterns while penalizing unstable over-fragmentation and physically broad clusters.

#### 6. Fallback and finalization

If no HDBSCAN candidate is accepted, deterministic KMeans models are compared using silhouette score.

For the selected model:

- one observed trajectory medoid is chosen per cluster;
- HDBSCAN `-1` trajectories are retained in the cluster artifact for provenance;
- those `-1` trajectories are rejected from the operational traffic corpus;
- consequently, rejected outliers do not appear in corpus visualization or scenario generation.

This produces runway-specific results from a shared global search policy while preserving deterministic and auditable model selection.

### How to use
In order to visualize the clustering results, use the following script (but before running it, you should run the build_offline_corpus code first (see section 3) to generate the corpus of good trajectories that `hailmary` will be able to replay when scaling demand).
To build the trajectory corpus, run the following command:
```bash
OUT=data/artifacts/hailmary/corpus
./.venv/bin/python -m hailmary.cli.build_offline_corpus \
  --manifest data_manifest.json --airport KDFW --output-dir "$OUT"
```
then to visualize:
```bash
./.venv/bin/python -m hailmary.clustering.visualization \
  --corpus-dir data/artifacts/hailmary/corpus \
  --manifest data_manifest.json
```

## 3. Building ADS-B trajectory corpus and visualize the results
A trajectory corpus is a set of observed ADS-B trajectories that will be used to scale traffic demand in Hail Mary scenarios. The corpus only contains trajectories that are considered to be valid, such as terminating "properly" at the runway threshold (the exact definition is quite nuanced to account for edge cases like flight number and icao24 are designated for both arrival and imminent departure) For example: if a window `00:20-01:20` has 30 traffic counts, then a scale of 1.1 will create 3 additional traffic counts. That means that 3 flights will be pulled from the corpus for the corresponding (runway arrival) cluster.

To build the trajectory corpus, run the following command:
```bash
OUT=data/artifacts/hailmary/corpus
./.venv/bin/python -m hailmary.cli.build_offline_corpus \
  --manifest data_manifest.json --airport KDFW --output-dir "$OUT"
```

Then we can visualize the corpus by running the script:
```bash
./.venv/bin/python src/hailmary/cli/visualize_offline_corpus.py \
  --corpus-dir data/artifacts/hailmary/corpus \
  --manifest data_manifest.json
```

## 4. Demand scaling algorithm 

The idea is that it will try to "inject" more flights into the current schedule according to the `scale` parameter given. For example: a scale of 2.0 would mean **all sliding windows of 60 minutes will observe the total flight counts equal two times their previous values accordingly.**

Here is a concrete run using the current KDFW offline corpus, scale `2.0`, seed `17`, and UTC times.

| Period | W0: 09:00–10:00 | W1: 09:20–10:20 |
|---|---:|---:|
| 09:00–09:20, W0-exclusive | 1 observed + 2 synthetic | — |
| 09:20–10:00, common | 6 observed + 5 synthetic | 6 observed + 7 synthetic |
| 10:00–10:20, W1-exclusive | — | 9 observed + 8 synthetic |
| Whole-window total | 7 → 14 | 15 → 30 |

### W0: 09:00–10:00

The builder found 7 observed flights and targeted 14.

It generated:

- 2 synthetic flights in the exclusive 09:00–09:20 period, at approximately `09:05:41` and `09:13:23`.
- 5 synthetic flights in the common 09:20–10:00 period.

Therefore:

```text
W0 = 1 observed exclusive
   + 6 observed common
   + 2 synthetic exclusive
   + 5 synthetic common
   = 14 flights
```

### W1: 09:20–10:20

The builder independently found 15 observed flights and targeted 30.

It generated:

- 7 synthetic flights in the common 09:20–10:00 period.
- 8 synthetic flights in the exclusive 10:00–10:20 period.

Therefore:

```text
W1 = 6 observed common
   + 9 observed exclusive
   + 7 synthetic common
   + 8 synthetic exclusive
   = 30 flights
```

### What happened to the common period

The six real flights in 09:20–10:00 were included in both scenarios. They retained the same observed IDs:

```text
AAL1005M1a07522
AAL2159M1a3661f
ATN762M1a37245
FDX1471M1a304d8
FDX1723M1a0bd33
FFT1650M1a3d7dc
```

But the synthetic flights were generated separately:

```text
shared observed IDs between W0 and W1: 6
shared synthetic IDs between W0 and W1: 0
```

Even inside the common period, W0’s five synthetic flights and W1’s seven synthetic flights are different scenario-specific flights.

### Why the two common-period synthetic counts differ

The code does not scale each 20- or 40-minute subperiod separately. Its effective process is:

```python
W0_baseline = all arrivals where 09:00 <= time < 10:00
W1_baseline = all arrivals where 09:20 <= time < 10:20

for each window independently:
    for each runway-cluster:
        target = round_half_up(scale * observed_count)
        generate target - observed_count additions
        sample each addition's time anywhere inside that window
```

The filtering happens in [`build_scenario()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/scenario/traffic.py:601), while synthetic times are sampled in [`_sample_time()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/scenario/traffic.py:510).

So the code does not explicitly reason about:

```text
previous-window-exclusive
common
current-window-exclusive
```

Those categories only emerge afterward from where the independently sampled timestamps happen to fall. The two windows are complete, independent simulation snapshots—not sections of a single coherent scaled timeline.

No code was changed for this example. 

The “intensity pool” is the historical set of arrivals used only to decide plausible times of day for synthetic flights. It does not determine how many flights are added—the scale calculation fixes that count first.

### 1. Count first, time second

Suppose window W1 is `[09:20, 10:20)` and cluster `KDFW:RW17C:1` contains three observed flights:

```text
09:28
09:44
10:12
```

With scale `2.0`:

```text
observed = 3
target   = round_half_up(3 × 2.0) = 6
additions required = 6 − 3 = 3
```

The code must therefore produce exactly three synthetic flights. It then samples a timestamp separately for each addition.

### 2. Selecting the intensity pool

For the cluster being scaled, [`_intensity_pool()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/scenario/traffic.py:492) uses this fallback hierarchy:

```text
same airport + same runway + same cluster
             ↓ if fewer than 4 historical arrivals
same airport + same runway, all clusters
             ↓ if fewer than 4 historical arrivals
same airport, all runways and clusters
```

The default minimum is four arrivals:

```python
sparse_cluster_min_count = 4
```

For example, suppose the training corpus contains:

| Airport | Runway | Cluster | Historical terminal-entry times |
|---|---|---|---|
| KDFW | RW17C | C1 | 08:55, 09:30, 09:50, 10:10, 13:00, 18:00 |
| KDFW | RW17C | C2 | 09:40, 10:02, 14:20 |
| KDFW | RW18R | C3 | 09:35, 10:05 |

For `KDFW:RW17C:C1`, there are six historical arrivals, so the exact cluster pool is used:

```text
pool = [08:55, 09:30, 09:50, 10:10, 13:00, 18:00]
scope = "cluster"
```

If C1 had only two historical arrivals, but RW17C had at least four total arrivals, the pool would become:

```text
pool = all RW17C arrivals
scope = "runway"
```

If RW17C were also sparse, it would use every KDFW arrival and record:

```text
scope = "airport"
```

The selected scope is saved in each synthetic flight’s `intensity_scope` metadata.

By default, `training_arrivals` is the complete corpus supplied to the builder. A caller can provide a separate training-only corpus through [`TrafficScenarioBuilder(..., training_arrivals=...)`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/scenario/traffic.py:398). The current traffic-batch CLI does not provide one, so it uses the complete input corpus.

### 3. Sampling one synthetic time

For each addition, [`_sample_time()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/scenario/traffic.py:510) repeats the following:

```text
1. Uniformly choose one arrival from the intensity pool.
2. Take that arrival’s UTC time of day.
3. Add Gaussian noise with standard deviation 1,200 seconds.
4. Wrap the result around midnight if necessary.
5. Accept it if it falls inside the current one-hour window.
6. Otherwise try again, up to 256 attempts.
```

The default bandwidth is 1,200 seconds, or 20 minutes:

```python
intensity_bandwidth_s = 1200.0
```

This is equivalent to placing a 20-minute-wide bell curve around every historical arrival time and sampling from the combined curves.

Conceptually, the time density resembles:

```text
historical arrivals
       ↓
08:55       09:30   09:50   10:10               13:00       18:00
  /\          /\      /\      /\
 /  \        /  \    /  \    /  \
```

> The "chevrons" are the density bell curves. 

For a `[09:20, 10:20)` scenario, the sampler conditions this distribution on being inside that window. Times near `09:30`, `09:50`, and `10:10` are therefore much more likely than times far from historical activity.

### 4. Concrete sampling example

Continuing with three required additions:

#### Addition 0

The random generator selects historical center `09:50`.

```text
Gaussian offset: +8 minutes
candidate:       09:58
```

`09:58` is inside `[09:20, 10:20)`, so it is accepted.

This falls in the common period between the two rolling windows.

#### Addition 1

First attempt:

```text
selected center: 10:10
Gaussian offset: +15 minutes
candidate:       10:25
```

`10:25` is outside the window, so it is rejected.

Second attempt:

```text
selected center: 09:30
Gaussian offset: −5 minutes
candidate:       09:25
```

`09:25` is accepted and falls in the common period.

#### Addition 2

```text
selected center: 09:50
Gaussian offset: +25 minutes
candidate:       10:15
```

`10:15` is accepted and falls in W1’s exclusive period.

The resulting synthetic split is:

```text
09:20–10:00 common period:       2 additions
10:00–10:20 W1-exclusive period: 1 addition
```

The code never requested a `2/1` split. That split emerged from the sampled times.

If no candidate lands inside the window after 256 attempts, the fallback is a uniform timestamp anywhere in the window.

### 5. Mathematical interpretation

For historical times of day \(t_1,\ldots,t_N\), the implicit density is approximately:

$$
\hat f(t)=\frac{1}{N}\sum_{i=1}^{N}
\mathcal N_{\text{circular}}(t;t_i,h^2),
$$

where:

```text
h = intensity_bandwidth_s = 1,200 seconds
```

The sampler then conditions this density on the scenario window:

$$
t_{\text{synthetic}}\sim\hat f(t)\mid t\in[\text{window start},\text{window end}).
$$

The implementation does not explicitly calculate this formula; selecting a historical center and adding Gaussian noise produces the same kernel-mixture sampling behavior.

### 6. Intensity pool versus donor pool

These are separate:

- The intensity pool decides the synthetic flight’s time.
- The donor pool supplies its ground speed, altitude, baseline variant, and provenance.

The donor is preferably selected from the exact same cluster, even when the time intensity had to fall back to runway or airport level. This happens in [`_materialize_cluster()`](/Volumes/CrucialX/project-rustlingtree/src/hailmary/scenario/traffic.py:552).

For example:

```text
Time sampled using all RW17C arrivals
Donor selected specifically from KDFW:RW17C:C1
```

Therefore, a sparse cluster may borrow its traffic timing pattern from its runway while retaining the cluster’s trajectory and flight-state characteristics.

Finally, the randomness is reproducible. Each addition receives a deterministic random stream derived from the master seed, dataset, window, cluster, scale, replicate, and addition index. Same inputs produce the same centers, offsets, donors, and timestamps. Different replicates produce alternative realizations with the same exact target counts.