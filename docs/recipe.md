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