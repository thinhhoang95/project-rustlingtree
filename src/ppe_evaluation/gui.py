from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from plotly.offline import get_plotlyjs

from ppe_evaluation.artifacts import (
    GroundTruth,
    MedoidTrajectory,
    default_ground_truth_dir,
    load_ground_truth,
    load_run_artifacts,
    medoid_station_frame,
    save_ground_truth,
)


INTERVENTION_CLASSES = ["no_stretch", "dogleg", "trombone", "PMS", "other"]


def create_app(run_dir: str | Path, ground_truth_dir: str | Path | None = None) -> FastAPI:
    predictions = load_run_artifacts(run_dir)
    gt_dir = Path(ground_truth_dir) if ground_truth_dir is not None else default_ground_truth_dir(predictions.run_dir)
    app = FastAPI(title="PPE Ground Truth Grader")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html()

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/plotly.js")
    def plotly_js() -> Response:
        return Response(get_plotlyjs(), media_type="application/javascript")

    @app.get("/api/run")
    def run_payload() -> JSONResponse:
        return JSONResponse(_run_payload(predictions, gt_dir))

    @app.get("/api/ground-truth")
    def ground_truth_payload() -> JSONResponse:
        if not (gt_dir / "manifest.json").exists():
            return JSONResponse({"exists": False, "accepted_cluster_ids": [], "windows": []})
        return JSONResponse(_ground_truth_payload(gt_dir))

    @app.post("/api/save")
    async def save_payload(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            saved = _save_from_payload(payload, predictions, gt_dir)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"status": "saved", "ground_truth_dir": saved.as_posix()})

    return app


def run_gui(
    run_dir: str | Path,
    ground_truth_dir: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    import uvicorn

    app = create_app(run_dir, ground_truth_dir)
    uvicorn.run(app, host=host, port=port, reload=False)


def _run_payload(predictions, gt_dir: Path) -> dict[str, Any]:
    medoids = []
    traces_by_cluster = _adsb_traces_by_cluster(predictions)
    for medoid in predictions.medoids:
        stations = medoid_station_frame(medoid)
        cluster_id = int(medoid.cluster_id) if medoid.cluster_id is not None else None
        medoids.append(
            {
                "cluster_id": cluster_id,
                "medoid_id": medoid.medoid_id,
                "medoid_track_id": medoid.medoid_track_id,
                "points": medoid.points.tolist(),
                "stations": stations.to_dict("records"),
                "traces": traces_by_cluster.get(cluster_id, []),
            }
        )
    return {
        "run_id": predictions.run_id,
        "dataset_id": predictions.dataset_id,
        "ground_truth_dir": gt_dir.as_posix(),
        "intervention_classes": INTERVENTION_CLASSES,
        "medoids": medoids,
        "predicted_windows": _predicted_windows(predictions.windows),
    }


def _adsb_traces_by_cluster(predictions) -> dict[int, list[dict[str, Any]]]:
    resampled_path = _state_path(
        predictions.state,
        "resampled_tracks_path",
        predictions.run_dir / "processed" / "resampled_tracks.parquet",
    )
    assignments_path = _state_path(
        predictions.state,
        "cluster_assignments_path",
        predictions.run_dir / "templates" / "chosen_cluster_assignments.csv",
    )
    if not resampled_path.exists() or not assignments_path.exists():
        return {}

    resampled = pd.read_parquet(resampled_path)
    assignments = pd.read_csv(assignments_path)
    required_tracks = {"flight_id", "station_index", "x_nm", "y_nm"}
    required_assignments = {"flight_id", "cluster_id"}
    if not required_tracks.issubset(resampled.columns) or not required_assignments.issubset(assignments.columns):
        return {}

    resampled = resampled.copy()
    assignments = assignments.copy()
    resampled["flight_id"] = resampled["flight_id"].astype(str)
    assignments["flight_id"] = assignments["flight_id"].astype(str)
    assignments["cluster_id"] = assignments["cluster_id"].astype(int)
    cluster_by_flight = assignments.set_index("flight_id")["cluster_id"].to_dict()

    traces: dict[int, list[dict[str, Any]]] = {}
    for flight_id, group in resampled.groupby("flight_id", sort=False):
        cluster_id = cluster_by_flight.get(str(flight_id))
        if cluster_id is None:
            continue
        ordered = group.sort_values("station_index", kind="stable")
        traces.setdefault(int(cluster_id), []).append(
            {
                "flight_id": str(flight_id),
                "points": ordered[["x_nm", "y_nm"]].astype(float).values.tolist(),
            }
        )
    return traces


def _state_path(state: dict[str, Any], key: str, default: Path) -> Path:
    value = state.get(key)
    return Path(value) if value else default


def _predicted_windows(windows) -> list[dict[str, Any]]:
    if windows.empty:
        return []
    columns = [
        "cluster_id",
        "window_id",
        "class_name",
        "confidence",
        "start_station_index",
        "end_station_index",
        "start_s_fraction",
        "end_s_fraction",
        "start_s_nm",
        "end_s_nm",
    ]
    available = [column for column in columns if column in windows.columns]
    return windows[available].to_dict("records")


def _ground_truth_payload(gt_dir: Path) -> dict[str, Any]:
    truth = load_ground_truth(gt_dir)
    medoid_cluster_by_key = {
        (str(medoid.medoid_id), int(medoid.cluster_id) if medoid.cluster_id is not None else None): medoid.cluster_id
        for medoid in truth.medoids
    }
    clusters_by_id: dict[str, list[int]] = {}
    for medoid in truth.medoids:
        if medoid.cluster_id is not None:
            clusters_by_id.setdefault(str(medoid.medoid_id), []).append(int(medoid.cluster_id))
    windows = []
    for row in truth.windows.to_dict("records"):
        payload = dict(row)
        gt_medoid_id = str(row["gt_medoid_id"])
        source_cluster_id = _source_cluster_id_from_window_id(str(row.get("window_id", "")))
        payload["cluster_id"] = medoid_cluster_by_key.get((gt_medoid_id, source_cluster_id))
        if payload["cluster_id"] is None and len(clusters_by_id.get(gt_medoid_id, [])) == 1:
            payload["cluster_id"] = clusters_by_id[gt_medoid_id][0]
        windows.append(payload)
    return {
        "exists": True,
        "accepted_cluster_ids": [
            medoid.cluster_id for medoid in truth.medoids if medoid.cluster_id is not None
        ],
        "windows": windows,
    }


def _save_from_payload(payload: dict[str, Any], predictions, gt_dir: Path) -> Path:
    accepted_cluster_ids = {int(item) for item in payload.get("accepted_cluster_ids", [])}
    medoid_by_cluster = {
        int(medoid.cluster_id): medoid for medoid in predictions.medoids if medoid.cluster_id is not None
    }
    selected_clusters = sorted(cluster_id for cluster_id in accepted_cluster_ids if cluster_id in medoid_by_cluster)

    existing_ids: dict[int, str] = {}
    if (gt_dir / "manifest.json").exists():
        existing = load_ground_truth(gt_dir)
        existing_ids = {
            int(medoid.cluster_id): medoid.medoid_id
            for medoid in existing.medoids
            if medoid.cluster_id is not None
        }

    gt_medoids: list[MedoidTrajectory] = []
    gt_id_by_cluster: dict[int, str] = {}
    for index, cluster_id in enumerate(selected_clusters):
        source = medoid_by_cluster[cluster_id]
        gt_id = existing_ids.get(cluster_id, f"GT{index:03d}")
        gt_id_by_cluster[cluster_id] = gt_id
        gt_medoids.append(
            MedoidTrajectory(
                medoid_id=gt_id,
                points=source.points,
                cluster_id=cluster_id,
                medoid_track_id=source.medoid_track_id,
                source_run_id=predictions.run_id,
            )
        )

    window_rows: list[dict[str, Any]] = []
    counters: dict[int, int] = {}
    for item in payload.get("windows", []):
        cluster_id = int(item["cluster_id"])
        if cluster_id not in gt_id_by_cluster:
            continue
        medoid = medoid_by_cluster[cluster_id]
        stations = medoid_station_frame(medoid)
        max_station = int(stations["station_index"].max())
        start = max(0, min(max_station, int(item["start_station_index"])))
        end = max(0, min(max_station, int(item["end_station_index"])))
        if end < start:
            start, end = end, start
        start_row = stations.loc[stations["station_index"] == start].iloc[0]
        end_row = stations.loc[stations["station_index"] == end].iloc[0]
        counters[cluster_id] = counters.get(cluster_id, 0) + 1
        window_rows.append(
            {
                "gt_medoid_id": gt_id_by_cluster[cluster_id],
                "window_id": str(item.get("window_id") or f"C{cluster_id}_GTW{counters[cluster_id]}"),
                "class_name": _valid_class(str(item.get("class_name") or "other")),
                "start_station_index": start,
                "end_station_index": end,
                "start_s_fraction": float(start_row["s_fraction"]),
                "end_s_fraction": float(end_row["s_fraction"]),
                "start_s_nm": float(start_row["s_nm"]),
                "end_s_nm": float(end_row["s_nm"]),
                "notes": str(item.get("notes") or ""),
            }
        )

    import pandas as pd

    manifest = {
        "seed_run_id": predictions.run_id,
        "source_run_dir": predictions.run_dir.as_posix(),
        "coordinate_system": predictions.state.get("coordinate_system"),
    }
    save_ground_truth(
        GroundTruth(
            dataset_id=predictions.dataset_id,
            medoids=gt_medoids,
            windows=pd.DataFrame(window_rows),
            manifest=manifest,
        ),
        gt_dir,
    )
    return gt_dir


def _source_cluster_id_from_window_id(window_id: str) -> int | None:
    if not window_id.startswith("C"):
        return None
    digits = []
    for char in window_id[1:]:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else None


def _valid_class(value: str) -> str:
    return value if value in INTERVENTION_CLASSES else "other"


def _html() -> str:
    class_options = "\n".join(f'<option value="{item}">{item}</option>' for item in INTERVENTION_CLASSES)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PPE Ground Truth Grader</title>
  <script src="/plotly.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; }}
    #app {{ display: grid; grid-template-columns: 320px 1fr; height: 100vh; }}
    aside {{ border-right: 1px solid #d9dee7; padding: 12px; overflow: auto; background: #f7f9fc; }}
    main {{ display: grid; grid-template-rows: 1fr auto; min-width: 0; }}
    #plot {{ width: 100%; height: 100%; }}
    .row {{ display: flex; gap: 8px; align-items: center; }}
    .medoid {{ padding: 8px; border-bottom: 1px solid #e2e8f0; cursor: pointer; }}
    .medoid.current {{ background: #e7f0ff; }}
    .track {{ font-size: 12px; color: #52606d; overflow-wrap: anywhere; }}
    button, select {{ font: inherit; }}
    button {{ border: 1px solid #bac5d5; background: white; padding: 6px 9px; border-radius: 6px; cursor: pointer; }}
    button.primary {{ background: #1f6feb; color: white; border-color: #1f6feb; }}
    #windowPanel {{ border-top: 1px solid #d9dee7; padding: 10px 12px; background: white; max-height: 220px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e5eaf0; padding: 5px; }}
    .hint {{ color: #66788a; font-size: 12px; margin: 8px 0; }}
  </style>
</head>
<body>
<div id="app">
  <aside>
    <div class="row">
      <button id="save" class="primary">Save</button>
      <select id="className">{class_options}</select>
    </div>
    <div id="status" class="hint"></div>
    <div class="hint">Gray ADS-B traces show the active cluster. Drag-select station markers on the active medoid to add a window.</div>
    <div id="medoids"></div>
  </aside>
  <main>
    <div id="plot"></div>
    <div id="windowPanel">
      <table>
        <thead><tr><th>Cluster</th><th>Window</th><th>Class</th><th>Stations</th><th></th></tr></thead>
        <tbody id="windows"></tbody>
      </table>
    </div>
  </main>
</div>
<script>
let runData = null;
let accepted = new Set();
let currentCluster = null;
let windowsByCluster = {{}};

function byClusterId(clusterId) {{
  return runData.medoids.find((m) => Number(m.cluster_id) === Number(clusterId));
}}

function predictedFor(clusterId) {{
  return runData.predicted_windows
    .filter((w) => Number(w.cluster_id) === Number(clusterId))
    .map((w, index) => ({{
      cluster_id: Number(clusterId),
      window_id: w.window_id || `C${{clusterId}}_W${{index + 1}}`,
      class_name: w.class_name || "other",
      start_station_index: Number(w.start_station_index),
      end_station_index: Number(w.end_station_index),
      notes: ""
    }}));
}}

function ensureWindows(clusterId) {{
  if (!windowsByCluster[clusterId]) {{
    windowsByCluster[clusterId] = predictedFor(clusterId);
  }}
}}

function renderMedoids() {{
  const root = document.getElementById("medoids");
  root.innerHTML = "";
  runData.medoids.forEach((m) => {{
    const div = document.createElement("div");
    div.className = "medoid" + (Number(m.cluster_id) === Number(currentCluster) ? " current" : "");
    div.onclick = () => {{ currentCluster = Number(m.cluster_id); render(); }};
    const checked = accepted.has(Number(m.cluster_id)) ? "checked" : "";
    div.innerHTML = `<label><input type="checkbox" ${{checked}}> Cluster ${{m.cluster_id}}</label>
      <div class="track">${{m.medoid_track_id || ""}}</div>`;
    div.querySelector("input").onclick = (event) => {{
      event.stopPropagation();
      const clusterId = Number(m.cluster_id);
      if (event.target.checked) {{
        accepted.add(clusterId);
        ensureWindows(clusterId);
        currentCluster = clusterId;
      }} else {{
        accepted.delete(clusterId);
      }}
      render();
    }};
    root.appendChild(div);
  }});
}}

function renderPlot() {{
  const traces = [];
  const current = byClusterId(currentCluster);
  if (current) {{
    (current.traces || []).forEach((trace) => {{
      traces.push({{
        x: trace.points.map((p) => p[0]),
        y: trace.points.map((p) => p[1]),
        mode: "lines",
        name: trace.flight_id,
        line: {{ width: 0.8, color: "#9aa7b7" }},
        opacity: 0.28,
        hoverinfo: "skip",
        showlegend: false
      }});
    }});
  }}
  runData.medoids.forEach((m) => {{
    const acceptedLine = accepted.has(Number(m.cluster_id));
    const isCurrent = Number(m.cluster_id) === Number(currentCluster);
    traces.push({{
      x: m.points.map((p) => p[0]),
      y: m.points.map((p) => p[1]),
      mode: "lines",
      name: `cluster ${{m.cluster_id}}`,
      line: {{ width: isCurrent ? 4 : (acceptedLine ? 2.5 : 1), color: isCurrent ? "#d97706" : (acceptedLine ? "#1f6feb" : "#9aa7b7") }},
      opacity: acceptedLine || isCurrent ? 1 : 0.35,
      hoverinfo: "name"
    }});
  }});
  if (current) {{
    traces.push({{
      x: current.stations.map((s) => s.x_nm),
      y: current.stations.map((s) => s.y_nm),
      mode: "markers",
      name: "stations",
      marker: {{ size: 7, color: "#111827" }},
      customdata: current.stations.map((s) => [s.station_index, Number(s.s_nm).toFixed(2)]),
      hovertemplate: "station %{{customdata[0]}}<br>s=%{{customdata[1]}} NM<extra></extra>"
    }});
    (windowsByCluster[currentCluster] || []).forEach((w) => {{
      const start = Math.min(w.start_station_index, w.end_station_index);
      const end = Math.max(w.start_station_index, w.end_station_index);
      const segment = current.stations.filter((s) => s.station_index >= start && s.station_index <= end);
      traces.push({{
        x: segment.map((s) => s.x_nm),
        y: segment.map((s) => s.y_nm),
        mode: "lines",
        name: `${{w.window_id}} ${{w.class_name}}`,
        line: {{ width: 8, color: "#dc2626" }},
        opacity: 0.72,
        hoverinfo: "name"
      }});
    }});
  }}
  Plotly.newPlot("plot", traces, {{
    margin: {{ l: 45, r: 15, t: 25, b: 45 }},
    xaxis: {{ title: "x (NM)", zeroline: false }},
    yaxis: {{ title: "y (NM)", scaleanchor: "x", scaleratio: 1, zeroline: false }},
    dragmode: "select",
    selectdirection: "any",
    showlegend: true
  }}, {{ responsive: true, displaylogo: false }});
  document.getElementById("plot").on("plotly_selected", (event) => {{
    if (!event || !event.points || currentCluster === null || !accepted.has(Number(currentCluster))) return;
    const stationIds = event.points
      .filter((point) => point.data && point.data.name === "stations")
      .map((point) => Number(point.customdata[0]));
    if (stationIds.length < 2) return;
    const start = Math.min(...stationIds);
    const end = Math.max(...stationIds);
    ensureWindows(currentCluster);
    const next = windowsByCluster[currentCluster].length + 1;
    windowsByCluster[currentCluster].push({{
      cluster_id: Number(currentCluster),
      window_id: `C${{currentCluster}}_GTW${{next}}`,
      class_name: document.getElementById("className").value,
      start_station_index: start,
      end_station_index: end,
      notes: ""
    }});
    render();
  }});
}}

function renderWindows() {{
  const body = document.getElementById("windows");
  body.innerHTML = "";
  Array.from(accepted).sort((a, b) => a - b).forEach((clusterId) => {{
    ensureWindows(clusterId);
    windowsByCluster[clusterId].forEach((w, index) => {{
      const row = document.createElement("tr");
      row.innerHTML = `<td>${{clusterId}}</td><td>${{w.window_id}}</td><td>${{w.class_name}}</td>
        <td>${{w.start_station_index}}-${{w.end_station_index}}</td>
        <td><button>Delete</button></td>`;
      row.querySelector("button").onclick = () => {{
        windowsByCluster[clusterId].splice(index, 1);
        render();
      }};
      body.appendChild(row);
    }});
  }});
}}

function render() {{
  renderMedoids();
  renderPlot();
  renderWindows();
}}

async function save() {{
  const allWindows = [];
  Array.from(accepted).forEach((clusterId) => {{
    (windowsByCluster[clusterId] || []).forEach((w) => allWindows.push(w));
  }});
  const response = await fetch("/api/save", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ accepted_cluster_ids: Array.from(accepted), windows: allWindows }})
  }});
  const payload = await response.json();
  document.getElementById("status").textContent = response.ok ? `Saved ${{payload.ground_truth_dir}}` : payload.detail;
}}

async function load() {{
  runData = await (await fetch("/api/run")).json();
  const truth = await (await fetch("/api/ground-truth")).json();
  accepted = new Set((truth.accepted_cluster_ids || []).map(Number));
  currentCluster = runData.medoids.length ? Number(runData.medoids[0].cluster_id) : null;
  windowsByCluster = {{}};
  if (truth.exists) {{
    (truth.windows || []).forEach((w) => {{
      const clusterId = Number(w.cluster_id);
      if (!windowsByCluster[clusterId]) windowsByCluster[clusterId] = [];
      windowsByCluster[clusterId].push({{
        cluster_id: clusterId,
        window_id: w.window_id,
        class_name: w.class_name,
        start_station_index: Number(w.start_station_index),
        end_station_index: Number(w.end_station_index),
        notes: w.notes || ""
      }});
    }});
  }}
  document.getElementById("status").textContent = `Run ${{runData.run_id}} -> ${{runData.ground_truth_dir}}`;
  render();
}}

document.getElementById("save").onclick = save;
load();
</script>
</body>
</html>"""
