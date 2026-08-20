"""Self-contained browser client for :mod:`visualize_event_queue`."""

from __future__ import annotations

EVENT_QUEUE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hailmary · Event Queue Verifier</title>
  <style>
    :root {
      --ink: #0b1620;
      --muted: #637483;
      --paper: #f3f6f4;
      --card: rgba(255, 255, 255, .92);
      --line: #d9e1dd;
      --navy: #092a3b;
      --cyan: #19b7b0;
      --lime: #b9d94b;
      --amber: #f2a93b;
      --red: #e76b62;
      --shadow: 0 14px 36px rgba(8, 34, 47, .09);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 7% 0%, rgba(25, 183, 176, .11), transparent 25rem),
        linear-gradient(180deg, #edf3f0 0, var(--paper) 28rem);
    }
    button, input { font: inherit; }
    .shell { width: min(1680px, calc(100% - 32px)); margin: 0 auto 40px; }
    header {
      display: flex; justify-content: space-between; gap: 24px; align-items: flex-end;
      padding: 26px 2px 18px;
    }
    .eyebrow { color: #178c88; font: 700 11px/1.2 ui-monospace, monospace; letter-spacing: .17em; text-transform: uppercase; }
    h1 { margin: 5px 0 3px; font-size: clamp(26px, 3vw, 42px); line-height: 1; letter-spacing: -.045em; }
    .subtitle { color: var(--muted); font-size: 14px; }
    .truth-badge {
      flex: none; display: flex; align-items: center; gap: 9px; padding: 9px 13px;
      border: 1px solid rgba(25, 183, 176, .3); border-radius: 999px;
      background: rgba(255, 255, 255, .72); color: #176a68; font: 700 11px/1 ui-monospace, monospace;
    }
    .truth-badge::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 0 4px rgba(25,183,176,.14); }
    .summary { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; margin-bottom: 10px; }
    .metric, .card { background: var(--card); border: 1px solid rgba(210, 221, 216, .95); box-shadow: var(--shadow); }
    .metric { min-height: 92px; border-radius: 14px; padding: 14px 16px; }
    .metric.wide { grid-column: span 2; }
    .metric .label { color: var(--muted); font: 700 10px/1.2 ui-monospace, monospace; text-transform: uppercase; letter-spacing: .11em; }
    .metric .value { display: block; margin-top: 7px; font-size: 27px; font-weight: 730; letter-spacing: -.04em; }
    .metric .value.window { font-size: 17px; letter-spacing: -.02em; line-height: 1.35; }
    .metric .minor { color: var(--muted); font-size: 11px; }
    .card { border-radius: 16px; overflow: hidden; }
    .card-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 14px 16px 12px; border-bottom: 1px solid var(--line); }
    .card-head h2 { margin: 0; font-size: 13px; letter-spacing: -.01em; }
    .micro { color: var(--muted); font: 600 10px/1.3 ui-monospace, monospace; }
    .top-grid { display: grid; grid-template-columns: minmax(0, 1.75fr) minmax(330px, .75fr); gap: 10px; }
    .radar { min-height: 530px; background: var(--navy); position: relative; }
    #radar { display: block; width: 100%; height: 530px; }
    .radar-overlay { position: absolute; left: 18px; top: 17px; color: #d8f4f0; pointer-events: none; }
    .radar-time { font: 700 18px/1 ui-monospace, monospace; letter-spacing: -.04em; }
    .radar-sub { margin-top: 6px; color: #78a7ae; font: 600 10px/1.3 ui-monospace, monospace; text-transform: uppercase; letter-spacing: .1em; }
    .legend { position: absolute; right: 14px; top: 14px; display: flex; gap: 8px; padding: 7px 9px; border: 1px solid rgba(160,205,207,.18); border-radius: 9px; background: rgba(6,25,36,.78); color: #a8c3c7; font: 600 9px/1 ui-monospace, monospace; }
    .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 3px; }
    .dot.scheduled { background: #77939b; } .dot.active { background: var(--lime); } .dot.completed { background: #2b6f77; }
    .flight-panel { display: flex; flex-direction: column; min-height: 530px; max-height: 530px; }
    .flight-list { overflow: auto; flex: 1; }
    .flight-row { display: grid; grid-template-columns: 8px minmax(90px, 1fr) auto; gap: 10px; align-items: center; padding: 11px 15px; border-bottom: 1px solid #edf1ef; }
    .status-bar { width: 5px; height: 31px; border-radius: 3px; background: #8ba0a7; }
    .status-bar.active { background: var(--lime); } .status-bar.completed { background: #34828a; }
    .flight-name { font: 740 12px/1.2 ui-monospace, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .flight-meta { margin-top: 3px; color: var(--muted); font-size: 10px; }
    .flight-data { text-align: right; font: 650 10px/1.45 ui-monospace, monospace; }
    .synthetic { color: #af6e00; }
    .timeline-card { margin-top: 10px; padding: 16px; }
    .timeline-top { display: flex; align-items: center; gap: 12px; }
    .controls { display: flex; gap: 7px; flex: none; }
    .controls button {
      border: 1px solid #bfd0ca; background: #fff; color: var(--ink); border-radius: 9px;
      min-height: 38px; padding: 0 13px; cursor: pointer; font-weight: 700; font-size: 12px;
    }
    .controls button.primary { border-color: var(--navy); background: var(--navy); color: #fff; }
    .controls button:disabled { opacity: .35; cursor: default; }
    .scrubber { min-width: 0; flex: 1; }
    input[type="range"] { width: 100%; accent-color: var(--cyan); cursor: pointer; }
    .timeline-labels { display: flex; justify-content: space-between; gap: 14px; margin-top: 8px; color: var(--muted); font: 600 10px/1.3 ui-monospace, monospace; }
    .timeline-labels .current { color: #126f6c; font-weight: 800; text-align: center; }
    .batch-strip { display: flex; align-items: center; gap: 7px; min-height: 30px; margin-top: 12px; padding-top: 11px; border-top: 1px solid var(--line); overflow-x: auto; }
    .batch-label { flex: none; color: var(--muted); font: 700 9px/1 ui-monospace, monospace; letter-spacing: .1em; text-transform: uppercase; }
    .chip { flex: none; border-radius: 999px; padding: 5px 8px; color: #245260; background: #e8f3f1; font: 700 9px/1 ui-monospace, monospace; }
    .chip.exogenous { background: #fff0d9; color: #9a5b00; }
    .chip.completed { background: #e7eceb; color: #5f7478; }
    .bottom-grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(360px, .65fr); gap: 10px; margin-top: 10px; }
    .queue-card, .actions-card { min-height: 480px; max-height: 620px; display: flex; flex-direction: column; }
    .badge { border-radius: 999px; padding: 4px 8px; background: #e9efec; color: #53656c; font: 700 9px/1 ui-monospace, monospace; }
    .table-wrap { overflow: auto; flex: 1; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    th { position: sticky; top: 0; z-index: 1; padding: 9px 11px; text-align: left; background: #f7f9f8; color: var(--muted); border-bottom: 1px solid var(--line); font: 700 9px/1 ui-monospace, monospace; text-transform: uppercase; letter-spacing: .08em; }
    td { padding: 9px 11px; border-bottom: 1px solid #edf1ef; vertical-align: top; }
    tbody tr:hover { background: #f6faf8; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .kind { display: inline-block; border-radius: 5px; padding: 3px 5px; background: #e9f5f2; color: #147772; font: 750 9px/1 ui-monospace, monospace; }
    .kind.EXOGENOUS_DISTURBANCE { background: #fff0d9; color: #9a5b00; }
    .kind.FLIGHT_COMPLETED { background: #e8edeb; color: #5f7478; }
    .action-content { overflow: auto; flex: 1; padding: 13px 15px 16px; }
    .section-label { margin: 2px 0 9px; color: var(--muted); font: 750 9px/1 ui-monospace, monospace; letter-spacing: .1em; text-transform: uppercase; }
    .vocabulary { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 17px; }
    .vocab { border: 1px solid #d5e2dd; border-radius: 7px; padding: 6px 8px; background: #f8faf9; font: 700 9px/1 ui-monospace, monospace; }
    .action { border: 1px solid #dbe5e1; border-left: 4px solid var(--cyan); border-radius: 10px; padding: 11px 12px; margin-bottom: 8px; }
    .action.no_op { border-left-color: #9baba8; }
    .action-title { display: flex; justify-content: space-between; gap: 8px; font: 760 11px/1.25 ui-monospace, monospace; }
    .action-detail { margin-top: 7px; color: var(--muted); font-size: 10px; line-height: 1.45; }
    .empty { margin-top: 7px; padding: 19px 16px; border: 1px dashed #cbd8d3; border-radius: 10px; color: var(--muted); font-size: 11px; line-height: 1.55; }
    .footnote { margin-top: 10px; padding: 12px 15px; color: #64757b; font-size: 10px; line-height: 1.55; text-align: center; }
    .loading { height: 100vh; display: grid; place-items: center; color: var(--muted); font: 700 12px/1 ui-monospace, monospace; }
    @media (max-width: 1050px) {
      .summary { grid-template-columns: repeat(3, 1fr); }
      .metric.wide { grid-column: span 2; }
      .top-grid, .bottom-grid { grid-template-columns: 1fr; }
      .flight-panel { min-height: 360px; max-height: 360px; }
    }
    @media (max-width: 680px) {
      .shell { width: min(100% - 18px, 1680px); }
      header { align-items: flex-start; flex-direction: column; }
      .summary { grid-template-columns: repeat(2, 1fr); }
      .metric.wide { grid-column: span 2; }
      .timeline-top { align-items: stretch; flex-direction: column; }
      .controls button { flex: 1; }
      #radar, .radar { height: 430px; min-height: 430px; }
    }
  </style>
</head>
<body>
  <div id="loading" class="loading">RECONSTRUCTING QUEUE STATES…</div>
  <div id="app" class="shell" hidden>
    <header>
      <div>
        <div class="eyebrow">Hailmary / deterministic verifier</div>
        <h1>Event Queue Radar</h1>
        <div id="subtitle" class="subtitle"></div>
      </div>
      <div class="truth-badge">PRODUCTION QUEUE REPLAY</div>
    </header>

    <section class="summary">
      <div class="metric wide"><span class="label">Intervention window</span><span id="window" class="value window"></span><span id="scale" class="minor"></span></div>
      <div class="metric"><span class="label">Original count</span><span id="original-count" class="value"></span><span class="minor">observed arrivals</span></div>
      <div class="metric"><span class="label">New flight count</span><span id="new-count" class="value"></span><span id="delta-count" class="minor"></span></div>
      <div class="metric"><span class="label">Event batches</span><span id="batch-count" class="value"></span><span id="event-count" class="minor"></span></div>
    </section>

    <section class="top-grid">
      <div class="card radar">
        <svg id="radar" role="img" aria-label="Flight position radar"></svg>
        <div class="radar-overlay"><div id="radar-time" class="radar-time"></div><div id="radar-sub" class="radar-sub"></div></div>
        <div class="legend"><span><i class="dot scheduled"></i>scheduled</span><span><i class="dot active"></i>active</span><span><i class="dot completed"></i>completed</span></div>
      </div>
      <div class="card flight-panel">
        <div class="card-head"><h2>Scaled flight list</h2><span id="flight-status" class="micro"></span></div>
        <div id="flight-list" class="flight-list"></div>
      </div>
    </section>

    <section class="card timeline-card">
      <div class="timeline-top">
        <div class="controls">
          <button id="previous">← Previous Event</button>
          <button id="next" class="primary">Next Event →</button>
        </div>
        <div class="scrubber">
          <input id="scrubber" type="range" min="0" value="0" step="1" aria-label="Event timeline">
          <div class="timeline-labels"><span id="timeline-start"></span><span id="timeline-current" class="current"></span><span id="timeline-end"></span></div>
        </div>
      </div>
      <div id="batch-strip" class="batch-strip"></div>
    </section>

    <section class="bottom-grid">
      <div class="card queue-card">
        <div class="card-head"><h2>Pending event queue</h2><span id="queue-count" class="badge"></span></div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>#</th><th>Time</th><th>Event kind</th><th>Flight / resource</th><th>Queue identity</th></tr></thead>
            <tbody id="queue-body"></tbody>
          </table>
        </div>
      </div>
      <div class="card actions-card">
        <div class="card-head"><h2>Available actions</h2><span id="action-count" class="badge"></span></div>
        <div class="action-content">
          <div class="section-label">Catalog vocabulary</div>
          <div id="vocabulary" class="vocabulary"></div>
          <div class="section-label">Eligible at selected batch</div>
          <div id="actions"></div>
        </div>
      </div>
    </section>
    <div id="footnote" class="card footnote"></div>
  </div>

  <script>
    const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
    const fmt = new Intl.NumberFormat("en-US");
    const shortId = value => value ? (value.length > 20 ? value.slice(0, 9) + "…" + value.slice(-7) : value) : "—";
    let trace, queueByFrame, frameIndex = 0;

    function compareEvents(aRef, bRef) {
      const a = trace.event_catalog[aRef].sort_key;
      const b = trace.event_catalog[bRef].sort_key;
      for (let i = 0; i < a.length; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
      }
      return 0;
    }

    function reconstructQueues() {
      const states = [];
      let queue = new Set(trace.initial_queue_refs);
      states.push([...queue].sort(compareEvents));
      for (let i = 1; i < trace.frames.length; i++) {
        const frame = trace.frames[i];
        frame.queue_removed_refs.forEach(ref => queue.delete(ref));
        frame.queue_added_refs.forEach(ref => queue.add(ref));
        states.push([...queue].sort(compareEvents));
      }
      return states;
    }

    function initializeSummary() {
      document.getElementById("subtitle").textContent = `Scenario ${trace.scenario_id} · definition ${shortId(trace.definition_hash)}`;
      document.getElementById("window").textContent = trace.window.label;
      document.getElementById("scale").textContent = `Scale ${trace.scale} · replicate ${trace.replicate}`;
      document.getElementById("original-count").textContent = fmt.format(trace.original_count);
      document.getElementById("new-count").textContent = fmt.format(trace.new_flight_count);
      const delta = trace.new_flight_count - trace.original_count;
      document.getElementById("delta-count").textContent = `${delta >= 0 ? "+" : ""}${delta} net · ${trace.synthetic_count} synthetic · ${trace.removed_count} removed`;
      document.getElementById("batch-count").textContent = fmt.format(trace.frames.length - 1);
      document.getElementById("event-count").textContent = `${fmt.format(Object.keys(trace.event_catalog).length)} scheduled event instances`;
      const scrubber = document.getElementById("scrubber");
      scrubber.max = trace.frames.length - 1;
      document.getElementById("timeline-start").textContent = trace.frames[0].time_label;
      document.getElementById("timeline-end").textContent = trace.frames.at(-1).time_label;
      document.getElementById("vocabulary").innerHTML = trace.action_vocabulary.map(a => `<span class="vocab">${escapeHtml(a.lever)} / ${escapeHtml(a.band)}</span>`).join("");
      const routeText = trace.route_graph_source ? `Route graph: ${trace.route_graph_source}` : "No route graph was available; contextual segment actions may be empty.";
      document.getElementById("footnote").textContent = `Each frame after the first is the immutable state returned by one Simulator.advance_next() call. Equal-time events remain one batch in Hailmary priority order. Aircraft positions use the simulator's live trajectory clock. ${routeText}`;
    }

    function project(x, y) {
      const b = trace.map.bounds;
      const width = Math.max(b.max_x - b.min_x, 1e-12);
      const height = Math.max(b.max_y - b.min_y, 1e-12);
      return [45 + (x - b.min_x) / width * 910, 565 - (y - b.min_y) / height * 520];
    }

    function radarGrid() {
      let grid = "";
      for (let x = 45; x <= 955; x += 91) grid += `<line x1="${x}" y1="45" x2="${x}" y2="565" stroke="#174556" stroke-width="1"/>`;
      for (let y = 45; y <= 565; y += 65) grid += `<line x1="45" y1="${y}" x2="955" y2="${y}" stroke="#174556" stroke-width="1"/>`;
      return grid;
    }

    function renderRadar(frame) {
      const svg = document.getElementById("radar");
      svg.setAttribute("viewBox", "0 0 1000 610");
      const routeGroups = new Map();
      trace.map.routes.forEach(route => {
        const key = route.cluster_id || route.flight_id;
        if (!routeGroups.has(key)) routeGroups.set(key, route);
      });
      const routes = [...routeGroups.values()].map(route => {
        const points = route.points.map(([x, y]) => project(x, y).join(",")).join(" ");
        return `<polyline points="${points}" fill="none" stroke="#2c6674" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round" opacity=".68"><title>${escapeHtml(route.runway)} · ${escapeHtml(route.cluster_id)}</title></polyline>`;
      }).join("");
      const previous = trace.frames[Math.max(0, frameIndex - 1)];
      const previousById = new Map(previous.positions.map(position => [position.flight_id, position]));
      const markers = frame.positions.map(position => {
        const [x, y] = project(position.x, position.y);
        const old = previousById.get(position.flight_id) || position;
        const [ox, oy] = project(old.x, old.y);
        const angle = Math.abs(x - ox) + Math.abs(y - oy) < .01 ? 0 : Math.atan2(y - oy, x - ox) * 180 / Math.PI + 90;
        const fill = position.lifecycle === "active" ? "#b9d94b" : position.lifecycle === "completed" ? "#367b84" : "#78939b";
        const stroke = position.synthetic ? "#f2a93b" : "#e6fbf6";
        const altitude = position.altitude_m == null ? "—" : `${Math.round(position.altitude_m)} m`;
        return `<g transform="translate(${x.toFixed(2)} ${y.toFixed(2)})" opacity="${position.lifecycle === "active" ? 1 : .72}">
          <g transform="rotate(${angle.toFixed(1)})"><path d="M0,-8 L5,7 L0,4 L-5,7 Z" fill="${fill}" stroke="${stroke}" stroke-width="1.2"/></g>
          <text x="9" y="-7" fill="${position.lifecycle === "active" ? "#eaffaa" : "#9bb8bd"}" font-family="ui-monospace,monospace" font-size="9" font-weight="700">${escapeHtml(position.callsign)}</text>
          <title>${escapeHtml(position.callsign)} · ${escapeHtml(position.lifecycle)} · ${altitude}</title>
        </g>`;
      }).join("");
      svg.innerHTML = `<rect width="1000" height="610" fill="#092a3b"/>${radarGrid()}<g>${routes}</g><g>${markers}</g>`;
      document.getElementById("radar-time").textContent = frame.time_label.split(" ").at(-1);
      document.getElementById("radar-sub").textContent = `${trace.map.coordinate_mode} projection · frame ${frame.index}/${trace.frames.length - 1}`;
    }

    function renderFlights(frame) {
      const positions = new Map(frame.positions.map(item => [item.flight_id, item]));
      const counts = {scheduled: 0, active: 0, completed: 0};
      const rows = trace.flights.map(flight => {
        const p = positions.get(flight.flight_id);
        const lifecycle = p?.lifecycle || "scheduled";
        counts[lifecycle]++;
        const altitude = p?.altitude_m == null ? "—" : `${Math.round(p.altitude_m)} m`;
        const remaining = p?.remaining_distance_m == null ? "—" : `${(p.remaining_distance_m / 1000).toFixed(1)} km`;
        return `<div class="flight-row">
          <span class="status-bar ${lifecycle}"></span>
          <div><div class="flight-name ${flight.synthetic ? "synthetic" : ""}">${escapeHtml(flight.callsign)}${flight.synthetic ? " ✦" : ""}</div><div class="flight-meta">${escapeHtml(flight.runway || "no runway")} · ${escapeHtml(lifecycle)} · ${escapeHtml(flight.release_label.split(" ").at(-1))}</div></div>
          <div class="flight-data">${altitude}<br>${remaining}</div>
        </div>`;
      });
      document.getElementById("flight-list").innerHTML = rows.join("");
      document.getElementById("flight-status").textContent = `${counts.active} active · ${counts.scheduled} scheduled · ${counts.completed} complete`;
    }

    function renderBatch(frame) {
      const strip = document.getElementById("batch-strip");
      const events = frame.processed_event_refs.map(ref => trace.event_catalog[ref]);
      if (!events.length) {
        strip.innerHTML = `<span class="batch-label">Current batch</span><span class="chip">INITIAL STATE · NOTHING PROCESSED</span>`;
        return;
      }
      strip.innerHTML = `<span class="batch-label">Processed together</span>` + events.map(event => {
        const klass = event.kind === "EXOGENOUS_DISTURBANCE" ? "exogenous" : event.kind === "FLIGHT_COMPLETED" ? "completed" : "";
        return `<span class="chip ${klass}" title="${escapeHtml(event.event_id)}">${escapeHtml(event.kind)} · ${escapeHtml(event.flight_id || event.resource_id || "global")}</span>`;
      }).join("");
    }

    function renderQueue(frame) {
      const refs = queueByFrame[frameIndex];
      document.getElementById("queue-count").textContent = `${fmt.format(refs.length)} pending`;
      document.getElementById("queue-body").innerHTML = refs.map((ref, index) => {
        const event = trace.event_catalog[ref];
        const subject = [event.flight_id, event.resource_id].filter(Boolean).join(" / ") || "global";
        const payload = JSON.stringify(event.payload);
        return `<tr title="${escapeHtml(payload)}">
          <td class="mono">${index + 1}</td>
          <td class="mono">${escapeHtml(event.time_label.split(" ").at(-1))}</td>
          <td><span class="kind ${escapeHtml(event.kind)}">${escapeHtml(event.kind)}</span></td>
          <td><strong>${escapeHtml(subject)}</strong>${event.station_index >= 0 ? `<br><span class="micro">station ${event.station_index}</span>` : ""}</td>
          <td class="mono" title="${escapeHtml(event.event_id)}">${escapeHtml(shortId(event.event_id))}<br><span class="micro">priority ${event.priority} · seq ${event.insertion_sequence}</span></td>
        </tr>`;
      }).join("");
    }

    function renderActions(frame) {
      const actions = frame.available_actions;
      document.getElementById("action-count").textContent = `${actions.length} eligible`;
      if (!actions.length) {
        document.getElementById("actions").innerHTML = `<div class="empty">No contextual action is eligible at this exact batch. Hailmary only enumerates candidates when an active follower crosses a matching action station inside a live leader–follower segment queue.</div>`;
        return;
      }
      document.getElementById("actions").innerHTML = actions.map(action => `<div class="action ${escapeHtml(action.lever)}">
        <div class="action-title"><span>${escapeHtml(action.lever)} / ${escapeHtml(action.band)}</span><span>${action.feasible ? "ELIGIBLE" : "INFEASIBLE"}</span></div>
        <div class="action-detail">${escapeHtml(action.leader_id)} → ${escapeHtml(action.follower_id)}<br>${escapeHtml(action.segment_id)} · ${escapeHtml(action.resource_id)} · station ${action.station_index} at ${(action.station_m / 1000).toFixed(1)} km</div>
      </div>`).join("");
    }

    function render(index) {
      frameIndex = Math.max(0, Math.min(trace.frames.length - 1, Number(index)));
      const frame = trace.frames[frameIndex];
      const scrubber = document.getElementById("scrubber");
      scrubber.value = frameIndex;
      document.getElementById("previous").disabled = frameIndex === 0;
      document.getElementById("next").disabled = frameIndex === trace.frames.length - 1;
      document.getElementById("timeline-current").textContent = `${frame.time_label} · event batch ${frameIndex}/${trace.frames.length - 1} · epoch ${frame.decision_epoch_index}`;
      renderRadar(frame);
      renderFlights(frame);
      renderBatch(frame);
      renderQueue(frame);
      renderActions(frame);
    }

    async function boot() {
      const response = await fetch("/api/trace");
      if (!response.ok) throw new Error(`Trace request failed (${response.status})`);
      trace = await response.json();
      queueByFrame = reconstructQueues();
      initializeSummary();
      document.getElementById("scrubber").addEventListener("input", event => render(event.target.value));
      document.getElementById("previous").addEventListener("click", () => render(frameIndex - 1));
      document.getElementById("next").addEventListener("click", () => render(frameIndex + 1));
      document.addEventListener("keydown", event => {
        if (event.key === "ArrowLeft") render(frameIndex - 1);
        if (event.key === "ArrowRight") render(frameIndex + 1);
      });
      document.getElementById("loading").hidden = true;
      document.getElementById("app").hidden = false;
      render(0);
    }

    boot().catch(error => {
      document.getElementById("loading").textContent = `FAILED TO LOAD VERIFIER: ${error.message}`;
    });
  </script>
</body>
</html>
"""


__all__ = ["EVENT_QUEUE_HTML"]
