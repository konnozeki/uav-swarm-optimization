"""Run one seed of every CP3 model and save one interactive HTML report."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re

import plotly.io as pio

from src.baselines import (
    JOCCCentralizedProjectedGradient,
    JOCCDistributedProjectedGradient,
    JOCCGradientConfig,
    LiteratureStaticThenTransition,
    R2CBufferedForceConfig,
    R2CBufferedVirtualForce,
    StaticThenTransition,
)
from src.datasets import (
    scenario_from_geographic_geojson,
    scenario_from_target_map_csv,
)
from src.experiments.transition_cases import ring_formation
from src.plotly_transition_visualization import build_transition_figure
from src.problem import DEFAULT_MIN_SEPARATION
from src.proposed import TransitionAwareGA
from src.reconfiguration import ReconfigurationProblem


ALGORITHM_CHOICES = (
    "static",
    "jocc-cpgs",
    "jocc-dpgs",
    "r2c-ise",
    "transition-aware",
)

ALGORITHM_LABELS = {
    "static": "Tối ưu đội hình trước, đường đi sau",
    "jocc-cpgs": "JOCC CPGS 2026",
    "jocc-dpgs": "JOCC DPGS 2026",
    "r2c-ise": "R2C-ISE AAAI-26",
    "transition-aware": "GA có xét quá trình chuyển đội hình",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Chạy một seed của các mô hình tái cấu hình và gom toàn bộ "
            "mô phỏng chuyển động vào một file HTML tự chứa."
        )
    )
    parser.add_argument("target_map")
    parser.add_argument(
        "--input-format",
        choices=["auto", "csv", "geographic-geojson"],
        default="auto",
    )
    parser.add_argument("--name", default="external_map")
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--margin-m", type=float, default=1000.0)
    parser.add_argument("--n-uavs", type=int, default=6)
    parser.add_argument("--sensing-radius", type=float)
    parser.add_argument("--communication-radius", type=float)
    parser.add_argument(
        "--min-separation",
        type=float,
        default=DEFAULT_MIN_SEPARATION,
    )
    parser.add_argument("--max-speed", type=float, default=40.0)
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-radius", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--domain-iterations", type=int, default=80)
    parser.add_argument("--r2c-iterations", type=int, default=400)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=list(ALGORITHM_CHOICES),
        help="Mặc định chạy cả năm mô hình.",
    )
    parser.add_argument(
        "--subframes",
        type=int,
        default=1,
        help=(
            "Số khung hình nội suy giữa hai bước; mặc định 1 để file của năm "
            "mô hình không quá lớn."
        ),
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=90,
        help="Số mili giây cho mỗi khung hình.",
    )
    parser.add_argument(
        "--output",
        default="results/cp3_final/fema_all_models.html",
    )
    return parser.parse_args()


def make_algorithms(args):
    algorithms = {
        "static": StaticThenTransition(
            population_size=args.population,
            generations=args.generations,
        ),
        "jocc-cpgs": LiteratureStaticThenTransition(
            JOCCCentralizedProjectedGradient(
                JOCCGradientConfig(iterations=args.domain_iterations)
            )
        ),
        "jocc-dpgs": LiteratureStaticThenTransition(
            JOCCDistributedProjectedGradient(
                JOCCGradientConfig(
                    iterations=args.domain_iterations,
                    step_fraction=0.05,
                    connectivity_weight=0.80,
                )
            )
        ),
        "r2c-ise": LiteratureStaticThenTransition(
            R2CBufferedVirtualForce(
                R2CBufferedForceConfig(iterations=args.r2c_iterations)
            )
        ),
        "transition-aware": TransitionAwareGA(
            population_size=args.population,
            generations=args.generations,
        ),
    }
    return [(key, algorithms[key]) for key in args.algorithms]


def load_problem(args):
    map_path = Path(args.target_map)
    input_format = args.input_format
    if input_format == "auto":
        input_format = (
            "geographic-geojson"
            if map_path.suffix.lower() in {".geojson", ".json"}
            else "csv"
        )

    if input_format == "geographic-geojson":
        scenario, _ = scenario_from_geographic_geojson(
            map_path,
            name=args.name,
            n_uavs=args.n_uavs,
            sensing_radius=args.sensing_radius or 3000.0,
            communication_radius=args.communication_radius or 7000.0,
            min_separation=args.min_separation,
            margin_m=args.margin_m,
        )
    else:
        if args.width is None or args.height is None:
            raise ValueError("CSV input requires --width and --height")
        scenario = scenario_from_target_map_csv(
            map_path,
            name=args.name,
            width=args.width,
            height=args.height,
            n_uavs=args.n_uavs,
            sensing_radius=args.sensing_radius or 175.0,
            communication_radius=args.communication_radius or 280.0,
            min_separation=args.min_separation,
        )

    start_x = scenario.width / 2 if args.start_x is None else args.start_x
    start_y = scenario.height / 2 if args.start_y is None else args.start_y
    start_radius = (
        0.45 * scenario.communication_radius
        if args.start_radius is None
        else args.start_radius
    )
    start = ring_formation(
        (start_x, start_y),
        args.n_uavs,
        start_radius,
    )
    return ReconfigurationProblem(
        name=f"{args.name}_reconfiguration",
        scenario=scenario,
        start_positions=start,
        max_speed=args.max_speed,
        allow_reassignment=True,
    )


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", value)


def metric_row(label, algorithm_name, metrics, runtime):
    return {
        "label": label,
        "algorithm": algorithm_name,
        "coverage": metrics.weighted_coverage_ratio,
        "time": metrics.formation_time_sec,
        "travel": metrics.total_travel_distance,
        "joint": metrics.joint_fitness,
        "feasible": metrics.feasible,
        "runtime": runtime,
    }


def format_number(value, digits=3):
    if value != value:  # NaN
        return "—"
    return f"{value:.{digits}f}"


def build_report(title, rows, panels):
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{escape(row['label'])}</td>"
            f"<td>{format_number(row['coverage'])}</td>"
            f"<td>{format_number(row['time'], 1)}</td>"
            f"<td>{format_number(row['travel'], 1)}</td>"
            f"<td>{format_number(row['joint'])}</td>"
            f"<td>{'Có' if row['feasible'] else 'Không'}</td>"
            f"<td>{format_number(row['runtime'], 2)}</td>"
            "</tr>"
        )

    buttons = []
    panel_html = []
    for index, (key, label, content) in enumerate(panels):
        active = " active" if index == 0 else ""
        buttons.append(
            f'<button class="tab-button{active}" data-panel="{safe_id(key)}">'
            f"{escape(label)}</button>"
        )
        panel_html.append(
            f'<section id="panel-{safe_id(key)}" class="plot-panel{active}">'
            f"{content}</section>"
        )

    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f4f7fb; color: #172033; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 22px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .hint {{ margin: 0 0 18px; color: #566276; }}
    .card {{ background: white; border: 1px solid #dce3ee; border-radius: 12px;
             box-shadow: 0 4px 18px rgba(25, 45, 80, .06); overflow: hidden; }}
    .table-wrap {{ overflow-x: auto; margin-bottom: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e7ecf3; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #edf3fa; white-space: nowrap; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 12px; }}
    .tab-button {{ border: 1px solid #bcc9da; background: white; color: #28384e;
                   border-radius: 9px; padding: 9px 13px; cursor: pointer; }}
    .tab-button.active {{ background: #1769aa; border-color: #1769aa; color: white; }}
    .plot-panel {{ display: none; min-height: 680px; }}
    .plot-panel.active {{ display: block; }}
    .plot-panel .plotly-graph-div {{ min-height: 680px; }}
    @media (max-width: 700px) {{ main {{ padding: 12px; }} .plot-panel {{ min-height: 540px; }} }}
  </style>
</head>
<body>
<main>
  <h1>{escape(title)}</h1>
  <p class="hint">Chọn mô hình bên dưới, rồi dùng nút Play/Pause hoặc thanh thời gian trong biểu đồ.</p>
  <div class="card table-wrap">
    <table>
      <thead><tr><th>Mô hình</th><th>Bao phủ</th><th>Thời gian (s)</th>
      <th>Quãng đường (m)</th><th>Điểm chung</th><th>Hợp lệ</th><th>Thời gian chạy (s)</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </div>
  <nav class="tabs">{''.join(buttons)}</nav>
  <div class="card">{''.join(panel_html)}</div>
</main>
<script>
  document.querySelectorAll('.tab-button').forEach(button => {{
    button.addEventListener('click', () => {{
      document.querySelectorAll('.tab-button, .plot-panel').forEach(item => item.classList.remove('active'));
      button.classList.add('active');
      const panel = document.getElementById('panel-' + button.dataset.panel);
      panel.classList.add('active');
      const plot = panel.querySelector('.plotly-graph-div');
      if (plot && window.Plotly) window.Plotly.Plots.resize(plot);
    }});
  }});
</script>
</body>
</html>
"""


def main():
    args = parse_args()
    if args.subframes <= 0:
        raise ValueError("--subframes must be positive")
    if args.frame_ms <= 0:
        raise ValueError("--frame-ms must be positive")

    problem = load_problem(args)
    rows = []
    panels = []
    include_plotly = True

    for key, algorithm in make_algorithms(args):
        label = ALGORITHM_LABELS[key]
        print(f"Đang chạy {label} (seed={args.seed}) ...", flush=True)
        solution, runtime = algorithm.solve(problem, seed=args.seed)
        metrics = solution.evaluation
        rows.append(metric_row(label, algorithm.name, metrics, runtime))

        print(
            f"  coverage={metrics.weighted_coverage_ratio:.3f} | "
            f"time={metrics.formation_time_sec:.1f}s | "
            f"joint={metrics.joint_fitness:.3f} | "
            f"feasible={int(metrics.feasible)} | runtime={runtime:.2f}s"
        )

        if solution.transition_solution is None:
            content = "<p style='padding:24px'>Mô hình không tạo được đường đi.</p>"
        else:
            transition_problem = problem.transition_problem(
                solution.final_positions
            )
            figure = build_transition_figure(
                transition_problem,
                solution.transition_solution,
                title=f"{label} | seed={args.seed}",
                targets=problem.scenario.targets,
                sensing_radius=problem.scenario.sensing_radius,
                target_weights=problem.scenario.target_weights,
                subframes_per_step=args.subframes,
                frame_duration_ms=args.frame_ms,
            )
            content = pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs="inline" if include_plotly else False,
                auto_play=False,
                config={"displaylogo": False, "responsive": True},
                div_id=f"trajectory-{safe_id(key)}",
            )
            include_plotly = False
        panels.append((key, label, content))

    output = Path(args.output)
    if output.suffix.lower() != ".html":
        raise ValueError("--output must end with .html")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(
        f"So sánh đường đi: {args.name} | seed={args.seed}",
        rows,
        panels,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    print(f"\nĐã tạo file: {output.resolve()}")


if __name__ == "__main__":
    main()
