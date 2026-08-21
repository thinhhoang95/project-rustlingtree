"""Self-contained browser client for :mod:`visualize_route_graph`."""

ROUTE_GRAPH_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hailmary route-graph verifier</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071019;
      --panel: #0d1924;
      --panel-2: #111f2c;
      --line: #263746;
      --text: #e7eef5;
      --muted: #91a4b5;
      --cyan: #62d9ff;
      --shared: #ffbf47;
      --merge: #ff6b7a;
      --runway: #a98cff;
      --ok: #62dda5;
      --warn: #ff846e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at 70% -20%, #17334a 0, var(--bg) 42%);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      padding: 20px 24px 16px;
      border-bottom: 1px solid var(--line);
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 23px; letter-spacing: -.02em; }
    h2 { font-size: 15px; }
    h3 { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
    .subtitle { margin-top: 4px; color: var(--muted); }
    .hash { max-width: 420px; color: var(--muted); font: 11px ui-monospace, monospace; word-break: break-all; text-align: right; }
    .summary {
      display: grid;
      grid-template-columns: repeat(6, minmax(105px, 1fr));
      gap: 9px;
      padding: 14px 24px;
    }
    .metric, .card {
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    .metric { padding: 10px 12px; }
    .metric strong { display: block; font-size: 21px; line-height: 1.1; }
    .metric span { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
    .metric.shared strong { color: var(--shared); }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 390px;
      gap: 12px;
      padding: 0 24px 24px;
      min-height: calc(100vh - 172px);
    }
    .map-card { min-height: 670px; overflow: hidden; position: relative; }
    .toolbar {
      min-height: 58px;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    label { color: var(--muted); font-size: 12px; }
    select, button {
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid #354b5e;
      border-radius: 6px;
      padding: 6px 9px;
      font: inherit;
    }
    select { max-width: 280px; }
    button { cursor: pointer; }
    button:hover { border-color: var(--cyan); }
    #map { width: 100%; height: 610px; display: block; cursor: grab; touch-action: none; }
    #map.dragging { cursor: grabbing; }
    .map-bg { fill: #08131d; }
    .grid { stroke: #132432; stroke-width: 1; }
    .segment { fill: none; stroke: var(--cyan); stroke-width: 2.4; opacity: .78; cursor: pointer; vector-effect: non-scaling-stroke; }
    .segment:hover, .segment.selected { stroke: #e9f9ff; stroke-width: 5; opacity: 1; }
    .segment.shared { stroke: var(--shared); stroke-width: 7; opacity: .96; }
    .segment.shared:hover, .segment.shared.selected { stroke: #fff1bf; stroke-width: 10; }
    .node { vector-effect: non-scaling-stroke; stroke: #071019; stroke-width: 2; }
    .node.corridor { fill: var(--cyan); }
    .node.merge { fill: var(--merge); }
    .node.runway_endpoint { fill: var(--runway); }
    .legend {
      position: absolute;
      left: 14px;
      bottom: 13px;
      display: flex;
      gap: 13px;
      padding: 8px 10px;
      background: #071019d9;
      border: 1px solid var(--line);
      border-radius: 7px;
      color: var(--muted);
      font-size: 11px;
      pointer-events: none;
    }
    .swatch { display: inline-block; width: 19px; margin-right: 5px; vertical-align: 3px; border-top: 3px solid var(--cyan); }
    .swatch.shared { border-color: var(--shared); border-width: 7px; }
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; background: var(--merge); }
    .side { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
    .card-head { padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .card-body { padding: 13px 14px; }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }
    .tag {
      display: inline-block;
      margin: 3px 4px 0 0;
      padding: 2px 6px;
      border-radius: 999px;
      background: #182b39;
      color: #cfe9f7;
      font-size: 11px;
    }
    .tag.shared { color: #1d1504; background: var(--shared); font-weight: 700; }
    .facts { display: grid; grid-template-columns: 115px 1fr; gap: 6px 9px; margin-top: 10px; }
    .facts dt { color: var(--muted); }
    .facts dd { margin: 0; min-width: 0; }
    .gate { margin-top: 8px; padding: 8px; background: #08131d; border-radius: 6px; }
    .route-list { max-height: 250px; overflow: auto; }
    .route-row, .segment-row {
      display: grid;
      align-items: center;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid #1d2c39;
    }
    .route-row { grid-template-columns: 27px minmax(0, 1fr) 68px; }
    .segment-row { grid-template-columns: 28px minmax(0, 1fr) 54px; cursor: pointer; }
    .segment-row:hover { color: var(--cyan); }
    .ordinal { color: var(--muted); font: 11px ui-monospace, monospace; }
    .badge { border-radius: 4px; padding: 2px 5px; font-size: 10px; text-align: center; background: #1c3140; }
    .badge.shared { background: #594116; color: #ffd987; }
    .fidelity { color: var(--muted); font-size: 12px; }
    .fidelity strong { color: var(--text); }
    .flow {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 8px;
      margin: 10px 0;
      color: var(--muted);
      text-align: center;
    }
    .flow span { padding: 7px; border: 1px solid var(--line); border-radius: 6px; }
    .flow b { color: var(--shared); font-size: 17px; }
    .warning { padding: 8px 10px; margin-top: 8px; border-left: 3px solid var(--warn); background: #321c1c; color: #ffc3b9; }
    .ok { color: var(--ok); }
    #tooltip {
      position: fixed;
      display: none;
      z-index: 5;
      max-width: 330px;
      padding: 7px 9px;
      border: 1px solid #486174;
      border-radius: 6px;
      background: #071019f2;
      box-shadow: 0 8px 24px #0008;
      pointer-events: none;
      font-size: 12px;
    }
    @media (max-width: 1050px) {
      .summary { grid-template-columns: repeat(3, 1fr); }
      main { grid-template-columns: 1fr; }
      .map-card { min-height: 590px; }
      #map { height: 530px; }
      .hash { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Hailmary route-graph verifier</h1>
      <p class="subtitle">Canonical static segments, merge corridors, and event-queue resource gates</p>
    </div>
    <div id="hash" class="hash"></div>
  </header>
  <section id="summary" class="summary"></section>
  <main>
    <section class="card map-card">
      <div class="toolbar">
        <label for="partition">Airport / runway</label><select id="partition"></select>
        <label for="cluster">Route cluster</label><select id="cluster"></select>
        <button id="reset-view" type="button">Reset view</button>
        <span class="muted">Drag to pan · wheel to zoom · click a segment</span>
      </div>
      <svg id="map" viewBox="0 0 1000 700" role="img" aria-label="Route graph map">
        <rect class="map-bg" width="1000" height="700"></rect>
        <g id="grid"></g>
        <g id="viewport"></g>
      </svg>
      <div class="legend">
        <span><i class="swatch"></i>exclusive segment</span>
        <span><i class="swatch shared"></i>Common traffic segment</span>
        <span><i class="dot"></i>merge boundary</span>
        <span>→ flight direction</span>
      </div>
    </section>
    <aside class="side">
      <section class="card">
        <div class="card-head"><h2>Segment inspection</h2></div>
        <div id="segment-detail" class="card-body muted">Select a segment on the map.</div>
      </section>
      <section class="card">
        <div class="card-head"><h2>Selected route traversal</h2></div>
        <div id="route-detail" class="card-body route-list muted">Choose a route cluster.</div>
      </section>
      <section class="card">
        <div class="card-head"><h2>Runtime interpretation</h2></div>
        <div id="fidelity" class="card-body fidelity"></div>
      </section>
      <section class="card">
        <div class="card-head"><h2>Coverage checks</h2></div>
        <div id="coverage" class="card-body"></div>
      </section>
    </aside>
  </main>
  <div id="tooltip"></div>
  <script>
  'use strict';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const state = { data: null, partition: '', cluster: '', selected: '', zoom: 1, panX: 0, panY: 0 };
  const el = id => document.getElementById(id);
  const svgNode = (name, attrs = {}) => {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  const compact = value => Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const escapeHtml = value => String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  const shortId = value => value.length > 19 ? value.slice(0, 9) + '…' + value.slice(-7) : value;

  function metric(value, label, css = '') {
    return `<div class="metric ${css}"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`;
  }

  function initialize(data) {
    state.data = data;
    el('hash').textContent = `${data.dataset_id} · ${data.artifact_content_hash}`;
    const s = data.summary;
    el('summary').innerHTML = [
      metric(s.partition_count, 'runway partitions'), metric(s.cluster_count, 'route clusters'),
      metric(s.segment_count, 'segments'), metric(s.shared_segment_count, 'common segments', 'shared'),
      metric(s.merge_node_count, 'merge nodes'), metric(s.observed_arrival_count, 'observed arrivals')
    ].join('');
    fillPartitionSelect();
    renderFidelity();
    renderCoverage();
    drawGrid();
    updatePartition();
  }

  function fillPartitionSelect() {
    const select = el('partition');
    select.innerHTML = '';
    state.data.partitions.forEach((item, index) => {
      const option = document.createElement('option');
      option.value = item.key;
      option.textContent = `${item.key} · ${item.cluster_count} routes · ${item.shared_segment_count} common`;
      select.append(option);
      if (index === 0) state.partition = item.key;
    });
  }

  function updatePartition() {
    state.partition = el('partition').value || state.partition;
    const select = el('cluster');
    select.innerHTML = '<option value="">All route clusters</option>';
    routes().forEach(route => {
      const option = document.createElement('option');
      option.value = route.qualified_cluster_id;
      option.textContent = `${route.qualified_cluster_id} (${route.observed_arrival_count} observed)`;
      select.append(option);
    });
    state.cluster = '';
    state.selected = '';
    resetView();
    renderRoute();
    renderSegmentDetail();
  }

  function routes() {
    return state.data.routes.filter(item => `${item.airport}:${item.runway}` === state.partition);
  }
  function segments() {
    return state.data.segments.filter(item => `${item.airport}:${item.runway}` === state.partition);
  }

  function projection(items) {
    const points = items.flatMap(item => item.lat_deg.map((lat, index) => [item.lon_deg[index], lat]));
    if (!points.length) return () => [500, 350];
    const meanLat = points.reduce((sum, point) => sum + point[1], 0) / points.length;
    const cosine = Math.max(.2, Math.cos(meanLat * Math.PI / 180));
    const xs = points.map(point => point[0] * cosine);
    const ys = points.map(point => point[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const width = Math.max(maxX - minX, 1e-8), height = Math.max(maxY - minY, 1e-8);
    const scale = Math.min(890 / width, 600 / height);
    return (lon, lat) => [55 + (lon * cosine - minX) * scale, 650 - (lat - minY) * scale];
  }

  function renderMap() {
    const viewport = el('viewport');
    viewport.innerHTML = '';
    const items = segments();
    const project = projection(items);
    const visible = segment => !state.cluster || segment.cluster_ids.includes(state.cluster);
    [...items].sort((a, b) => Number(a.shared) - Number(b.shared)).forEach(segment => {
      const points = segment.lat_deg.map((lat, index) => project(segment.lon_deg[index], lat).join(',')).join(' ');
      const line = svgNode('polyline', {
        points, 'data-id': segment.segment_id,
        class: `segment${segment.shared ? ' shared' : ''}${segment.segment_id === state.selected ? ' selected' : ''}`,
        style: `opacity:${visible(segment) ? 1 : .07}`,
        'marker-end': segment.shared ? 'url(#arrow-shared)' : 'url(#arrow)'
      });
      line.addEventListener('click', event => { event.stopPropagation(); selectSegment(segment.segment_id); });
      line.addEventListener('pointermove', event => showTooltip(event, segment));
      line.addEventListener('pointerleave', hideTooltip);
      viewport.append(line);
    });
    const nodeIds = new Set(items.filter(visible).flatMap(item => [item.entry_node_id, item.exit_node_id]));
    state.data.nodes.filter(node => nodeIds.has(node.node_id)).forEach(node => {
      const [x, y] = project(node.lon_deg, node.lat_deg);
      let shape;
      if (node.kind === 'merge') {
        shape = svgNode('rect', { x: x - 3, y: y - 3, width: 6, height: 6, transform: `rotate(45 ${x} ${y})`, class: 'node merge' });
      } else if (node.kind === 'runway_endpoint') {
        shape = svgNode('rect', { x: x - 4, y: y - 4, width: 8, height: 8, class: 'node runway_endpoint' });
      } else {
        shape = svgNode('circle', { cx: x, cy: y, r: 3, class: 'node corridor' });
      }
      viewport.append(shape);
    });
    installMarkers(viewport);
    applyViewport();
  }

  function installMarkers(viewport) {
    const defs = svgNode('defs');
    [['arrow', '#62d9ff'], ['arrow-shared', '#ffbf47']].forEach(([id, color]) => {
      const marker = svgNode('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse', markerUnits: 'userSpaceOnUse' });
      marker.append(svgNode('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: color }));
      defs.append(marker);
    });
    viewport.prepend(defs);
  }

  function selectSegment(segmentId) {
    state.selected = segmentId;
    renderMap();
    renderSegmentDetail();
  }

  function renderSegmentDetail() {
    const target = el('segment-detail');
    const segment = state.data.segments.find(item => item.segment_id === state.selected);
    if (!segment) { target.className = 'card-body muted'; target.textContent = 'Select a segment on the map.'; return; }
    target.className = 'card-body';
    const tag = segment.shared ? '<span class="tag shared">COMMON TRAFFIC SEGMENT</span>' : '<span class="tag">exclusive segment</span>';
    const clusters = segment.cluster_ids.map(item => `<span class="tag">${escapeHtml(item)}</span>`).join('');
    target.innerHTML = `${tag}<h3 style="margin-top:10px">${escapeHtml(shortId(segment.segment_id))}</h3>
      <dl class="facts">
        <dt>Route clusters</dt><dd>${segment.cluster_count}</dd>
        <dt>Observed traffic</dt><dd>${compact(segment.observed_arrival_count)} arrivals</dd>
        <dt>Length</dt><dd>${compact(segment.length_nm)} NM</dd>
        <dt>Corridor width</dt><dd>${compact(segment.corridor_width_m)} m</dd>
      </dl>
      <div style="margin-top:8px">${clusters}</div>
      <div class="flow"><span>entry gate</span><b>→</b><span>exit gate / flow anchor</span></div>
      <div class="gate"><div class="muted">Entry resource</div><div class="mono">${escapeHtml(segment.entry_resource_id)}</div></div>
      <div class="gate"><div class="muted">Exit resource</div><div class="mono">${escapeHtml(segment.exit_resource_id)}</div></div>
      <details style="margin-top:9px"><summary>Full segment identity</summary><div class="mono" style="margin-top:6px">${escapeHtml(segment.segment_id)}</div></details>`;
  }

  function renderRoute() {
    const target = el('route-detail');
    const route = state.data.routes.find(item => item.qualified_cluster_id === state.cluster);
    if (!route) { target.className = 'card-body route-list muted'; target.textContent = 'Choose a route cluster to inspect its exact upstream-to-runway traversal order.'; return; }
    target.className = 'card-body route-list';
    const rows = route.traversals.map(item => `<div class="segment-row" data-segment="${escapeHtml(item.segment_id)}">
      <span class="ordinal">${item.ordinal}</span><span class="mono" title="${escapeHtml(item.segment_id)}">${escapeHtml(shortId(item.segment_id))}</span>
      <span class="badge ${item.shared ? 'shared' : ''}">${item.shared ? 'common' : 'exclusive'}</span></div>`).join('');
    target.innerHTML = `<h3>${escapeHtml(route.qualified_cluster_id)}</h3><p class="muted">${route.observed_arrival_count} observed arrivals · arrows point toward runway</p>${rows}`;
    target.querySelectorAll('[data-segment]').forEach(row => row.addEventListener('click', () => selectSegment(row.dataset.segment)));
  }

  function renderFidelity() {
    const f = state.data.fidelity;
    el('fidelity').innerHTML = `<p><strong>Sharing:</strong> ${escapeHtml(f.shared_segment_rule)}</p>
      <p style="margin-top:7px"><strong>Required interval:</strong> ${compact(state.data.config.required_interval_s)} seconds at every segment gate in this artifact.</p>
      <p style="margin-top:7px"><strong>Queue order:</strong> ${escapeHtml(f.runtime_queue_order)}</p>
      <p style="margin-top:7px"><strong>Resources:</strong> ${escapeHtml(f.queue_resource)}</p>`;
  }

  function renderCoverage() {
    const c = state.data.coverage, target = el('coverage');
    const warnings = [];
    if (!c.corpus_loaded) warnings.push('Traffic corpus was not loaded; observed traffic counts are unavailable.');
    if (c.corpus_clusters_missing_from_graph.length) warnings.push(`Corpus clusters missing from graph: ${c.corpus_clusters_missing_from_graph.join(', ')}`);
    if (c.graph_clusters_without_observed_traffic.length) warnings.push(`Graph clusters without observed corpus traffic: ${c.graph_clusters_without_observed_traffic.join(', ')}`);
    target.innerHTML = warnings.length ? warnings.map(item => `<div class="warning">${escapeHtml(item)}</div>`).join('') : '<span class="ok">✓ Every corpus cluster has a graph traversal and every graph cluster has observed traffic.</span>';
  }

  function showTooltip(event, segment) {
    const tip = el('tooltip');
    tip.style.display = 'block';
    tip.style.left = `${event.clientX + 14}px`; tip.style.top = `${event.clientY + 14}px`;
    tip.innerHTML = `<strong>${segment.shared ? 'Common' : 'Exclusive'} segment</strong><br>${segment.cluster_count} cluster${segment.cluster_count === 1 ? '' : 's'} · ${compact(segment.length_nm)} NM · ${segment.observed_arrival_count} observed arrivals`;
  }
  function hideTooltip() { el('tooltip').style.display = 'none'; }

  function drawGrid() {
    const grid = el('grid');
    for (let x = 100; x < 1000; x += 100) grid.append(svgNode('line', { x1: x, x2: x, y1: 0, y2: 700, class: 'grid' }));
    for (let y = 100; y < 700; y += 100) grid.append(svgNode('line', { x1: 0, x2: 1000, y1: y, y2: y, class: 'grid' }));
  }
  function applyViewport() { el('viewport').setAttribute('transform', `translate(${state.panX} ${state.panY}) scale(${state.zoom})`); }
  function resetView() { state.zoom = 1; state.panX = 0; state.panY = 0; renderMap(); }

  el('partition').addEventListener('change', updatePartition);
  el('cluster').addEventListener('change', event => { state.cluster = event.target.value; state.selected = ''; renderMap(); renderRoute(); renderSegmentDetail(); });
  el('reset-view').addEventListener('click', resetView);
  const map = el('map');
  let drag = null;
  map.addEventListener('pointerdown', event => { drag = { x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY }; map.setPointerCapture(event.pointerId); map.classList.add('dragging'); });
  map.addEventListener('pointermove', event => { if (!drag) return; const rect = map.getBoundingClientRect(); state.panX = drag.panX + (event.clientX - drag.x) * 1000 / rect.width; state.panY = drag.panY + (event.clientY - drag.y) * 700 / rect.height; applyViewport(); });
  map.addEventListener('pointerup', () => { drag = null; map.classList.remove('dragging'); });
  map.addEventListener('wheel', event => { event.preventDefault(); const rect = map.getBoundingClientRect(); const x = (event.clientX - rect.left) * 1000 / rect.width; const y = (event.clientY - rect.top) * 700 / rect.height; const old = state.zoom; state.zoom = Math.max(.5, Math.min(8, old * Math.exp(-event.deltaY * .001))); state.panX = x - (x - state.panX) * state.zoom / old; state.panY = y - (y - state.panY) * state.zoom / old; applyViewport(); }, { passive: false });
  map.addEventListener('click', () => { state.selected = ''; renderMap(); renderSegmentDetail(); });

  fetch('/api/route-graph').then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }).then(initialize).catch(error => {
    document.body.innerHTML = `<main style="padding:30px"><div class="warning">Could not load route graph: ${escapeHtml(error.message)}</div></main>`;
  });
  </script>
</body>
</html>
"""

__all__ = ["ROUTE_GRAPH_HTML"]
