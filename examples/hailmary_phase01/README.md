# Hailmary Phase 0/1 verification examples

Run these examples from the repository root with the locked local environment:

```bash
./.venv/bin/python examples/hailmary_phase01/01_demand_and_scaling.py
./.venv/bin/python examples/hailmary_phase01/02_route_graph_and_pairing.py
```

The scripts print canonical JSON. Checked-in copies of their output live in
[`results`](results), so a change is easy to review without starting the full
ADS-B build.

`01_demand_and_scaling.py` demonstrates half-open overlapping demand windows,
cluster-level half-up scaling, exact scale-one replay, and joint donor
provenance. `02_route_graph_and_pairing.py` builds converging routes and one
nearby parallel route, advances a real simulator, and shows how a live
speed-profile change rebuilds a segment-scoped queue while preserving every
`SegmentTraversalDefinition`. The topology builder itself is airport-wide and
can share multiple separated trunks across routes bound for different runways;
that join/split/rejoin case is covered by the topology tests and route-graph
verifier.

For the complete repository ADS-B day, run the public artifact chain in order:

```bash
OUT=/tmp/hailmary-phase01
./.venv/bin/python -m hailmary.cli.build_offline_corpus \
  --manifest data_manifest.json --airport KDFW --output-dir "$OUT"
./.venv/bin/python -m hailmary.cli.build_route_graph \
  --input "$OUT/route_graph_input.json" --output "$OUT/route_graph.json"
./.venv/bin/python -m hailmary.cli.build_traffic_batch \
  --corpus "$OUT/traffic_corpus.json" \
  --templates "$OUT/hailmary_templates.json" \
  --route-graph "$OUT/route_graph.json" \
  --output "$OUT/traffic_batch_scale1.json" \
  --start 1775022000 --stop 1775106000 --scale 1.0 \
  --source-partition 2026-04-01
./.venv/bin/python examples/hailmary_phase01/03_verify_real_pipeline.py "$OUT"
```

The recorded result shows the data-quality boundary explicitly: 1,026 catalog
arrivals produce 539 reconstructable terminal entries on eight runways, 16
templates, and 70 independent scale-one windows. It also reconstructs the real
10:20 UTC window and proves that its first reviewed cross-cluster pair reaches
a physical action epoch.
