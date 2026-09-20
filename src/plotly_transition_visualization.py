from __future__ import annotations

from pathlib import Path
import json

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .graph_ops import communication_graph_from_positions
from .obstacles import formation_obstacle_free
from .transition import (
    TransitionProblem,
    TransitionSolution,
    formation_min_distance,
)


def _covered_targets(
    positions: np.ndarray,
    targets: np.ndarray | None,
    sensing_radius: float | None,
) -> np.ndarray | None:
    """Return a binary target-coverage mask for visualization only."""
    if targets is None or sensing_radius is None:
        return None

    targets = np.asarray(targets, dtype=float)
    positions = np.asarray(positions, dtype=float)

    if len(targets) == 0:
        return np.zeros(0, dtype=bool)

    delta = targets[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    return np.any(distance <= sensing_radius, axis=1)


def _edge_coordinates(
    positions: np.ndarray,
    edges,
) -> tuple[list[float | None], list[float | None]]:
    """Convert graph edges to one Plotly multi-segment line trace."""
    xs: list[float | None] = []
    ys: list[float | None] = []

    for i, j in edges:
        xs.extend([
            float(positions[i, 0]),
            float(positions[j, 0]),
            None,
        ])
        ys.extend([
            float(positions[i, 1]),
            float(positions[j, 1]),
            None,
        ])

    return xs, ys


def _trail_coordinates(
    dense_trajectory: np.ndarray,
    frame_idx: int,
) -> tuple[list[float | None], list[float | None]]:
    """Combine all UAV trails into one lightweight Plotly line trace."""
    xs: list[float | None] = []
    ys: list[float | None] = []

    for uav_idx in range(dense_trajectory.shape[1]):
        path = dense_trajectory[: frame_idx + 1, uav_idx, :]

        xs.extend(path[:, 0].astype(float).tolist())
        xs.append(None)

        ys.extend(path[:, 1].astype(float).tolist())
        ys.append(None)

    return xs, ys


def _dense_trajectory(
    trajectory: np.ndarray,
    subframes_per_step: int,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Linearly interpolate recorded transition states for smooth browser motion.

    Returns
    -------
    dense:
        Interpolated trajectory with shape (K, N, 2).
    segment_indices:
        Original transition segment used by each dense frame. This is needed to
        display the corresponding certified backbone.
    time_fractions:
        Continuous time in units of original transition steps.
    """
    if subframes_per_step <= 0:
        raise ValueError("subframes_per_step must be positive")

    trajectory = np.asarray(trajectory, dtype=float)

    if len(trajectory) == 1:
        return (
            trajectory.copy(),
            [0],
            [0.0],
        )

    dense = []
    segment_indices = []
    time_fractions = []

    for step in range(len(trajectory) - 1):
        start = trajectory[step]
        end = trajectory[step + 1]

        # Do not include tau=1 here because it is tau=0 of the next segment.
        for subframe in range(subframes_per_step):
            tau = subframe / subframes_per_step
            dense.append(
                (1.0 - tau) * start + tau * end
            )
            segment_indices.append(step)
            time_fractions.append(step + tau)

    dense.append(trajectory[-1].copy())
    segment_indices.append(len(trajectory) - 1)
    time_fractions.append(float(len(trajectory) - 1))

    return (
        np.asarray(dense),
        segment_indices,
        time_fractions,
    )


def _frame_state(
    problem: TransitionProblem,
    solution: TransitionSolution,
    dense: np.ndarray,
    frame_idx: int,
    segment_idx: int,
    time_fraction: float,
    targets: np.ndarray | None,
    sensing_radius: float | None,
):
    """Build all dynamic traces plus one compact status annotation."""
    positions = dense[frame_idx]

    graph = communication_graph_from_positions(
        positions,
        problem.communication_radius,
    )
    communication_edges = list(graph.edges())
    comm_x, comm_y = _edge_coordinates(
        positions,
        communication_edges,
    )

    if segment_idx < len(solution.backbones):
        backbone_edges = solution.backbones[segment_idx]
    else:
        backbone_edges = ()

    backbone_x, backbone_y = _edge_coordinates(
        positions,
        backbone_edges,
    )

    trail_x, trail_y = _trail_coordinates(
        dense,
        frame_idx,
    )

    coverage_mask = _covered_targets(
        positions,
        targets,
        sensing_radius,
    )

    if (
        targets is not None
        and coverage_mask is not None
        and np.any(coverage_mask)
    ):
        covered = np.asarray(targets)[coverage_mask]
        covered_x = covered[:, 0]
        covered_y = covered[:, 1]
    else:
        covered_x = []
        covered_y = []

    connected = nx.is_connected(graph)
    min_distance = formation_min_distance(positions)
    obstacle_clear = formation_obstacle_free(
        positions,
        problem.obstacles,
        problem.obstacle_clearance,
    )
    elapsed = time_fraction * problem.dt

    if coverage_mask is None or len(coverage_mask) == 0:
        coverage_text = "n/a"
    else:
        coverage_text = f"{np.mean(coverage_mask):.1%}"

    status = (
        f"<b>t = {elapsed:.1f}s</b><br>"
        f"connected: {'yes' if connected else 'no'}<br>"
        f"obstacle clear: {'yes' if obstacle_clear else 'no'}<br>"
        f"min separation: {min_distance:.1f}<br>"
        f"coverage: {coverage_text}"
    )

    traces = [
        go.Scatter(
            x=covered_x,
            y=covered_y,
            mode="markers",
            marker={"size": 7, "opacity": 0.85},
            name="covered targets",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=comm_x,
            y=comm_y,
            mode="lines",
            line={"width": 1},
            opacity=0.35,
            name="communication",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=backbone_x,
            y=backbone_y,
            mode="lines",
            line={"width": 4},
            opacity=0.75,
            name="protected backbone",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=trail_x,
            y=trail_y,
            mode="lines",
            line={"width": 1.5},
            opacity=0.45,
            name="trajectory trail",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="markers+text",
            text=[str(i) for i in range(len(positions))],
            textposition="top right",
            marker={
                "size": 13,
                "color": list(range(len(positions))),
                "colorscale": "Viridis",
                "showscale": False,
                "line": {"width": 1, "color": "white"},
            },
            name="UAV",
            customdata=np.arange(len(positions)),
            hovertemplate=(
                "UAV %{customdata}<br>"
                "x=%{x:.1f}<br>"
                "y=%{y:.1f}<extra></extra>"
            ),
        ),
    ]

    annotation = {
        "x": 0.01,
        "y": 0.99,
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "top",
        "text": status,
        "showarrow": False,
        "align": "left",
        "bgcolor": "rgba(255,255,255,0.85)",
        "bordercolor": "rgba(0,0,0,0.15)",
        "borderwidth": 1,
        "borderpad": 6,
    }

    return traces, annotation


def _manual_editor_post_script(
    problem: TransitionProblem,
    solution: TransitionSolution,
    targets: np.ndarray | None,
    sensing_radius: float | None,
    target_weights: np.ndarray | None,
    goal_trace_index: int,
) -> str:
    """Return self-contained JS that makes formation-B UAV nodes draggable.

    The editor is intentionally browser-side only. It recomputes static
    formation diagnostics immediately while the user drags destination nodes,
    which is useful for visually demonstrating local improvements or failure
    cases without rerunning the Python optimizer.

    Transition time/travel shown by the editor are straight-line lower-bound
    estimates for the edited B, not a rerun of the obstacle-aware planner.
    """
    if targets is None:
        target_list = []
    else:
        target_list = np.asarray(
            targets,
            dtype=float,
        ).tolist()

    if target_weights is None:
        weight_list = [1.0] * len(target_list)
    else:
        weight_list = np.asarray(
            target_weights,
            dtype=float,
        ).tolist()

    obstacles = [
        {
            "x_min": float(obstacle.x_min),
            "y_min": float(obstacle.y_min),
            "x_max": float(obstacle.x_max),
            "y_max": float(obstacle.y_max),
        }
        for obstacle in problem.obstacles
    ]

    payload = {
        "starts": np.asarray(
            problem.start_positions,
            dtype=float,
        ).tolist(),
        "positions": np.asarray(
            solution.assigned_goals,
            dtype=float,
        ).tolist(),
        "targets": target_list,
        "weights": weight_list,
        "width": float(problem.width),
        "height": float(problem.height),
        "sensing_radius": (
            None
            if sensing_radius is None
            else float(sensing_radius)
        ),
        "communication_radius": float(
            problem.communication_radius
        ),
        "min_separation": float(
            problem.min_separation
        ),
        "max_speed": float(problem.max_speed),
        "obstacle_clearance": float(
            problem.obstacle_clearance
        ),
        "obstacles": obstacles,
        "goal_trace_index": int(goal_trace_index),
        # Match the current static objective defaults used by CP2/CP3.
        "redundancy_weight": 0.10,
        "connectivity_weight": 2.00,
        "collision_weight": 2.00,
    }

    payload_json = json.dumps(
        payload,
        separators=(",", ":"),
    )

    # Plotly replaces {plot_id} with the generated div id in post_script.
    return f"""
(function() {{
  const gd = document.getElementById('{{plot_id}}');
  const cfg = {payload_json};
  if (!gd || gd.__uavManualEditorInstalled) return;
  gd.__uavManualEditorInstalled = true;

  let positions = cfg.positions.map(p => [Number(p[0]), Number(p[1])]);
  const initialPositions = positions.map(p => p.slice());
  let activeIndex = null;
  let latestMetrics = null;
  let editing = false;

  const panel = document.createElement('div');
  panel.style.cssText = [
    'font-family:system-ui,sans-serif',
    'font-size:13px',
    'padding:8px 10px',
    'margin:0 0 8px 0',
    'border:1px solid #d9d9d9',
    'border-radius:8px',
    'background:#fff',
    'display:flex',
    'gap:12px',
    'align-items:center',
    'flex-wrap:wrap'
  ].join(';');

  const hint = document.createElement('span');
  hint.textContent = 'Edit B: drag a green X node. Metrics update live.';
  hint.style.fontWeight = '600';

  const metricsBox = document.createElement('span');
  metricsBox.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, monospace';

  const editButton = document.createElement('button');
  editButton.textContent = 'Edit B';
  editButton.type = 'button';

  const resetButton = document.createElement('button');
  resetButton.textContent = 'Reset B';
  resetButton.type = 'button';

  const copyButton = document.createElement('button');
  copyButton.textContent = 'Copy state';
  copyButton.type = 'button';

  for (const button of [editButton, resetButton, copyButton]) {{
    button.style.cssText = [
      'padding:5px 9px',
      'border:1px solid #bbb',
      'border-radius:6px',
      'background:#fafafa',
      'cursor:pointer'
    ].join(';');
  }}

  panel.appendChild(hint);
  panel.appendChild(metricsBox);
  panel.appendChild(editButton);
  panel.appendChild(resetButton);
  panel.appendChild(copyButton);
  gd.parentNode.insertBefore(panel, gd);

  function distance(a, b) {{
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return Math.hypot(dx, dy);
  }}

  function insideObstacle(p) {{
    const c = cfg.obstacle_clearance;
    return cfg.obstacles.some(o =>
      p[0] >= o.x_min - c &&
      p[0] <= o.x_max + c &&
      p[1] >= o.y_min - c &&
      p[1] <= o.y_max + c
    );
  }}

  function computeMetrics() {{
    const n = positions.length;
    const m = cfg.targets.length;
    const counts = new Array(m).fill(0);
    let totalWeight = 0;
    let coveredWeight = 0;
    let redundancyWeighted = 0;
    let potentialWeighted = 0;
    const decayScale = Math.max(
      0.75 * (cfg.sensing_radius || 1),
      1
    );

    for (let t = 0; t < m; t++) {{
      const target = cfg.targets[t];
      const weight = Number(cfg.weights[t] ?? 1);
      totalWeight += weight;
      let nearest = Infinity;

      for (let i = 0; i < n; i++) {{
        const d = distance(target, positions[i]);
        nearest = Math.min(nearest, d);
        if (
          cfg.sensing_radius !== null &&
          d <= cfg.sensing_radius + 1e-9
        ) {{
          counts[t] += 1;
        }}
      }}

      if (counts[t] > 0) coveredWeight += weight;
      if (counts[t] > 1) {{
        redundancyWeighted += weight * (counts[t] - 1);
      }}

      if (cfg.sensing_radius !== null) {{
        const gap = Math.max(
          nearest - cfg.sensing_radius,
          0
        );
        potentialWeighted += weight * Math.exp(
          -gap / decayScale
        );
      }}
    }}

    const denominator = Math.max(totalWeight, 1e-12);
    const coverage = m
      ? coveredWeight / denominator
      : 0;
    const redundancy = m
      ? redundancyWeighted / denominator
      : 0;
    const potential = m && cfg.sensing_radius !== null
      ? potentialWeighted / denominator
      : 0;

    const adjacency = Array.from(
      {{length:n}},
      () => []
    );
    let minSeparation = Infinity;
    let collisionViolations = 0;
    let communicationEdges = [];

    for (let i = 0; i < n; i++) {{
      for (let j = i + 1; j < n; j++) {{
        const d = distance(
          positions[i],
          positions[j]
        );
        minSeparation = Math.min(
          minSeparation,
          d
        );

        if (d <= cfg.communication_radius + 1e-9) {{
          adjacency[i].push(j);
          adjacency[j].push(i);
          communicationEdges.push([i, j]);
        }}

        if (d < cfg.min_separation - 1e-9) {{
          collisionViolations += 1;
        }}
      }}
    }}

    if (n <= 1) minSeparation = Infinity;

    const visited = new Array(n).fill(false);
    let components = 0;

    for (let seed = 0; seed < n; seed++) {{
      if (visited[seed]) continue;
      components += 1;
      const stack = [seed];
      visited[seed] = true;

      while (stack.length) {{
        const u = stack.pop();
        for (const v of adjacency[u]) {{
          if (!visited[v]) {{
            visited[v] = true;
            stack.push(v);
          }}
        }}
      }}
    }}

    const connected = n <= 1 || components === 1;
    const connectivityDeficit = n <= 1
      ? 0
      : (components - 1) / (n - 1);
    const pairCount = n * (n - 1) / 2;
    const collisionRatio = pairCount
      ? collisionViolations / pairCount
      : 0;
    const obstacleViolations = positions.filter(
      insideObstacle
    ).length;
    const obstacleClear = obstacleViolations === 0;

    const inBounds = positions.every(p =>
      p[0] >= 0 &&
      p[0] <= cfg.width &&
      p[1] >= 0 &&
      p[1] <= cfg.height
    );

    const staticFeasible = (
      inBounds &&
      connected &&
      collisionViolations === 0 &&
      obstacleClear
    );

    const staticFitness = (
      coverage
      - cfg.redundancy_weight * redundancy
      - cfg.connectivity_weight * connectivityDeficit
      - cfg.collision_weight * collisionRatio
    );

    let totalDirectTravel = 0;
    let maxDirectTravel = 0;

    for (let i = 0; i < n; i++) {{
      const d = distance(
        cfg.starts[i],
        positions[i]
      );
      totalDirectTravel += d;
      maxDirectTravel = Math.max(
        maxDirectTravel,
        d
      );
    }}

    const straightTimeLowerBound = (
      maxDirectTravel / Math.max(cfg.max_speed, 1e-12)
    );

    return {{
      coverage,
      potential,
      redundancy,
      connected,
      components,
      connectivityDeficit,
      minSeparation,
      collisionViolations,
      obstacleClear,
      obstacleViolations,
      inBounds,
      staticFeasible,
      staticFitness,
      totalDirectTravel,
      straightTimeLowerBound,
      communicationEdges,
      counts
    }};
  }}

  function edgeCoordinates(edges) {{
    const xs = [];
    const ys = [];

    for (const [i, j] of edges) {{
      xs.push(
        positions[i][0],
        positions[j][0],
        null
      );
      ys.push(
        positions[i][1],
        positions[j][1],
        null
      );
    }}

    return [xs, ys];
  }}

  function renderMetrics() {{
    const minSepText = Number.isFinite(
      latestMetrics.minSeparation
    )
      ? latestMetrics.minSeparation.toFixed(1)
      : 'inf';

    metricsBox.textContent = [
      'coverage=' + latestMetrics.coverage.toFixed(3),
      'potential=' + latestMetrics.potential.toFixed(3),
      'fitness=' + latestMetrics.staticFitness.toFixed(3),
      'connected=' + (latestMetrics.connected ? 'yes' : 'no'),
      'components=' + latestMetrics.components,
      'minSep=' + minSepText,
      'collisions=' + latestMetrics.collisionViolations,
      'obstacle=' + (latestMetrics.obstacleClear ? 'clear' : 'BLOCKED'),
      'directTravel=' + latestMetrics.totalDirectTravel.toFixed(1),
      'timeLB=' + latestMetrics.straightTimeLowerBound.toFixed(2) + 's'
    ].join(' | ');

    panel.style.borderColor = latestMetrics.staticFeasible
      ? '#9bc89b'
      : '#e09a9a';
  }}

  function refreshPlot() {{
    latestMetrics = computeMetrics();
    const xs = positions.map(p => p[0]);
    const ys = positions.map(p => p[1]);
    const [edgeX, edgeY] = edgeCoordinates(
      latestMetrics.communicationEdges
    );

    const coveredX = [];
    const coveredY = [];

    for (let t = 0; t < cfg.targets.length; t++) {{
      if (latestMetrics.counts[t] > 0) {{
        coveredX.push(cfg.targets[t][0]);
        coveredY.push(cfg.targets[t][1]);
      }}
    }}

    // Keep both the draggable B marker and the filled UAV marker on the edited
    // destination so the picture and the live metrics describe the same state.
    Plotly.restyle(
      gd,
      {{x:[xs], y:[ys]}},
      [cfg.goal_trace_index]
    );
    Plotly.restyle(
      gd,
      {{x:[xs], y:[ys]}},
      [5]
    );

    Plotly.restyle(
      gd,
      {{x:[coveredX], y:[coveredY]}},
      [1]
    );
    Plotly.restyle(
      gd,
      {{x:[edgeX], y:[edgeY]}},
      [2]
    );

    // These traces belong to the original planner trajectory and are no longer
    // valid after a manual destination edit.
    Plotly.restyle(
      gd,
      {{x:[[]], y:[[]]}},
      [3]
    );
    Plotly.restyle(
      gd,
      {{x:[[]], y:[[]]}},
      [4]
    );

    renderMetrics();
  }}

  function enterEditMode() {{
    if (editing) return;

    editing = true;
    editButton.textContent = 'Editing B';
    editButton.style.background = '#eef7ee';
    hint.textContent = (
      'Edit B: drag a numbered X/UAV node. ' +
      'Static metrics update live; directTravel/timeLB are straight-line estimates.'
    );

    Plotly.animate(
      gd,
      [null],
      {{
        mode:'immediate',
        frame:{{duration:0, redraw:false}},
        transition:{{duration:0}}
      }}
    );

    Plotly.relayout(
      gd,
      {{
        'sliders[0].visible': false,
        'updatemenus[0].visible': false
      }}
    );

    refreshPlot();
  }}

  function clientToData(event) {{
    const rect = gd.getBoundingClientRect();
    const xa = gd._fullLayout.xaxis;
    const ya = gd._fullLayout.yaxis;
    const px = (
      event.clientX
      - rect.left
      - xa._offset
    );
    const py = (
      event.clientY
      - rect.top
      - ya._offset
    );

    return [
      Number(xa.p2d(px)),
      Number(ya.p2d(py))
    ];
  }}

  function dataToClient(p) {{
    const rect = gd.getBoundingClientRect();
    const xa = gd._fullLayout.xaxis;
    const ya = gd._fullLayout.yaxis;

    return [
      rect.left + xa._offset + xa.d2p(p[0]),
      rect.top + ya._offset + ya.d2p(p[1])
    ];
  }}

  function nearestNode(event) {{
    let best = null;
    let bestDistance = 22;

    for (let i = 0; i < positions.length; i++) {{
      const [cx, cy] = dataToClient(
        positions[i]
      );
      const d = Math.hypot(
        event.clientX - cx,
        event.clientY - cy
      );

      if (d < bestDistance) {{
        best = i;
        bestDistance = d;
      }}
    }}

    return best;
  }}

  function pointerDown(event) {{
    if (!editing) return;

    const index = nearestNode(event);

    if (index === null) return;

    activeIndex = index;
    gd.style.cursor = 'grabbing';
    event.preventDefault();
    event.stopPropagation();
  }}

  function pointerMove(event) {{
    if (activeIndex === null) return;

    let [x, y] = clientToData(event);
    x = Math.max(0, Math.min(cfg.width, x));
    y = Math.max(0, Math.min(cfg.height, y));

    positions[activeIndex] = [x, y];
    refreshPlot();

    event.preventDefault();
    event.stopPropagation();
  }}

  function pointerUp(event) {{
    if (activeIndex === null) return;

    activeIndex = null;
    gd.style.cursor = '';
    event.preventDefault();
    event.stopPropagation();
  }}

  gd.addEventListener(
    'pointerdown',
    pointerDown,
    true
  );
  window.addEventListener(
    'pointermove',
    pointerMove,
    true
  );
  window.addEventListener(
    'pointerup',
    pointerUp,
    true
  );

  editButton.addEventListener('click', () => {{
    enterEditMode();
  }});

  resetButton.addEventListener('click', () => {{
    enterEditMode();
    positions = initialPositions.map(
      p => p.slice()
    );
    refreshPlot();
  }});

  copyButton.addEventListener('click', async () => {{
    const payload = {{
      positions: positions.map(p => [
        Number(p[0].toFixed(3)),
        Number(p[1].toFixed(3))
      ]),
      metrics: {{
        coverage: Number(
          latestMetrics.coverage.toFixed(6)
        ),
        potential: Number(
          latestMetrics.potential.toFixed(6)
        ),
        static_fitness: Number(
          latestMetrics.staticFitness.toFixed(6)
        ),
        connected: latestMetrics.connected,
        components: latestMetrics.components,
        min_separation: Number(
          latestMetrics.minSeparation.toFixed(6)
        ),
        collision_violations:
          latestMetrics.collisionViolations,
        obstacle_clear:
          latestMetrics.obstacleClear,
        total_direct_travel: Number(
          latestMetrics.totalDirectTravel.toFixed(3)
        ),
        straight_time_lower_bound_sec: Number(
          latestMetrics.straightTimeLowerBound.toFixed(3)
        )
      }}
    }};

    const text = JSON.stringify(
      payload,
      null,
      2
    );

    try {{
      await navigator.clipboard.writeText(text);
      copyButton.textContent = 'Copied';
      setTimeout(
        () => copyButton.textContent = 'Copy state',
        1200
      );
    }} catch (error) {{
      window.prompt(
        'Copy this state:',
        text
      );
    }}
  }});

  latestMetrics = computeMetrics();
  renderMetrics();
}})();
"""


def save_transition_html(
    problem: TransitionProblem,
    solution: TransitionSolution,
    output_path: str | Path,
    title: str | None = None,
    targets: np.ndarray | None = None,
    sensing_radius: float | None = None,
    target_weights: np.ndarray | None = None,
    subframes_per_step: int = 5,
    frame_duration_ms: int = 90,
    enable_manual_edit: bool = True,
) -> None:
    """Save an interactive browser visualization as one self-contained HTML.

    Plotly gives us a map-like 2D canvas with pan, zoom, hover, play/pause and a
    time slider. Recorded planner states are linearly interpolated only for
    display, so visualization remains smooth without changing optimization or
    transition metrics.

    The HTML embeds Plotly itself. It therefore opens offline and does not need
    a local web server.
    """
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".html":
        raise ValueError("interactive visualization output must be .html")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory = np.asarray(
        solution.trajectory,
        dtype=float,
    )

    dense, segment_indices, time_fractions = _dense_trajectory(
        trajectory,
        subframes_per_step,
    )

    if targets is not None:
        targets = np.asarray(targets, dtype=float)

        if target_weights is None:
            target_sizes = np.full(len(targets), 5.0)
        else:
            target_weights = np.asarray(
                target_weights,
                dtype=float,
            )
            normalized = target_weights / max(
                float(np.max(target_weights)),
                1e-12,
            )
            target_sizes = 4.0 + 5.0 * normalized
    else:
        target_sizes = []

    dynamic_traces, initial_annotation = _frame_state(
        problem,
        solution,
        dense,
        frame_idx=0,
        segment_idx=segment_indices[0],
        time_fraction=time_fractions[0],
        targets=targets,
        sensing_radius=sensing_radius,
    )

    if targets is None:
        target_x = []
        target_y = []
    else:
        target_x = targets[:, 0]
        target_y = targets[:, 1]

    static_target_trace = go.Scatter(
        x=target_x,
        y=target_y,
        mode="markers",
        marker={
            "size": target_sizes,
            "opacity": 0.25,
        },
        name="targets",
        hovertemplate="target<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
    )

    start_trace = go.Scatter(
        x=trajectory[0, :, 0],
        y=trajectory[0, :, 1],
        mode="markers",
        marker={
            "size": 14,
            "symbol": "circle-open",
            "line": {"width": 2},
        },
        name="formation A",
        hovertemplate="A<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
    )

    goal_trace = go.Scatter(
        x=solution.assigned_goals[:, 0],
        y=solution.assigned_goals[:, 1],
        mode="markers+text",
        text=[str(i) for i in range(problem.n_uavs)],
        textposition="top right",
        marker={
            "size": 14,
            "symbol": "x",
            "line": {"width": 2},
        },
        name="formation B",
        hovertemplate="B<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
    )

    # Trace order is fixed because animation frames update only indices 1..5.
    data = [
        static_target_trace,
        *dynamic_traces,
        start_trace,
        goal_trace,
    ]

    frames = []

    for frame_idx in range(len(dense)):
        traces, annotation = _frame_state(
            problem,
            solution,
            dense,
            frame_idx=frame_idx,
            segment_idx=segment_indices[frame_idx],
            time_fraction=time_fractions[frame_idx],
            targets=targets,
            sensing_radius=sensing_radius,
        )

        frames.append(
            go.Frame(
                name=str(frame_idx),
                data=traces,
                traces=[1, 2, 3, 4, 5],
                layout=go.Layout(
                    annotations=[annotation],
                ),
            )
        )

    if title is None:
        title = f"{problem.name} | {solution.planner}"

    slider_steps = [
        {
            "args": [
                [str(frame_idx)],
                {
                    "frame": {
                        "duration": 0,
                        "redraw": True,
                    },
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": (
                f"{time_fractions[frame_idx] * problem.dt:.1f}"
            ),
            "method": "animate",
        }
        for frame_idx in range(len(frames))
    ]

    fig = go.Figure(
        data=data,
        frames=frames,
    )

    obstacle_shapes = []

    for obstacle in problem.obstacles:
        obstacle_shapes.append(
            {
                "type": "rect",
                "x0": obstacle.x_min,
                "y0": obstacle.y_min,
                "x1": obstacle.x_max,
                "y1": obstacle.y_max,
                "fillcolor": "rgba(90,90,90,0.28)",
                "line": {
                    "color": "rgba(70,70,70,0.85)",
                    "width": 1.5,
                },
                "layer": "below",
            }
        )

        if problem.obstacle_clearance > 0:
            x_min, y_min, x_max, y_max = obstacle.expanded(
                problem.obstacle_clearance
            )
            obstacle_shapes.append(
                {
                    "type": "rect",
                    "x0": x_min,
                    "y0": y_min,
                    "x1": x_max,
                    "y1": y_max,
                    "fillcolor": "rgba(0,0,0,0)",
                    "line": {
                        "color": "rgba(90,90,90,0.45)",
                        "width": 1,
                        "dash": "dot",
                    },
                    "layer": "below",
                }
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 70, "b": 70},
        xaxis={
            "title": "x",
            "range": [0.0, problem.width],
            "constrain": "domain",
            "showgrid": True,
            "zeroline": False,
        },
        yaxis={
            "title": "y",
            "range": [0.0, problem.height],
            "scaleanchor": "x",
            "scaleratio": 1,
            "showgrid": True,
            "zeroline": False,
        },
        annotations=[initial_annotation],
        shapes=obstacle_shapes,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.12,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": frame_duration_ms,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.18,
                "y": -0.10,
                "len": 0.80,
                "currentvalue": {
                    "prefix": "t = ",
                    "suffix": " s",
                },
                "steps": slider_steps,
            }
        ],
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1.0,
        },
    )

    post_script = None

    if enable_manual_edit:
        post_script = _manual_editor_post_script(
            problem,
            solution,
            targets,
            sensing_radius,
            target_weights,
            goal_trace_index=len(data) - 1,
        )

    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
        post_script=post_script,
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )
