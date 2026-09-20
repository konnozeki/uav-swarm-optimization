# UAV Swarm Optimization — Checkpoint 02

Checkpoint này KHÔNG ghi đè Checkpoint 01.

Mục tiêu là giữ storyline phát triển:

```text
Checkpoint 01
model + metrics + toy baselines
        |
        v
Checkpoint 02
stronger baselines + graph-aware GA + ablation + statistics
```

## 1. Cài trên Windows

PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. Chạy demo

```powershell
python run_demo.py
```

Demo so:
- Vanilla GA
- GraphAwareGA

trên một scenario hai cụm target, nơi connectivity dễ trở thành nút thắt.

Ảnh được lưu vào:

```text
outputs/demo/
```

## 3. Benchmark nhanh

```powershell
python run_benchmark.py --profile quick --seeds 5
```

Các algorithm mặc định:

```text
random
greedy
ga
pso
nsga2
graph_ga
```

Kết quả:

```text
outputs/benchmark_quick/
    results.csv
    summary.csv
    figures/
```

## 4. Benchmark connectivity stress

```powershell
python run_benchmark.py --profile stress --seeds 10
```

Profile này thay đổi communication radius và số UAV để ép connectivity trở thành
bottleneck.

## 5. Full sweep

```powershell
python run_benchmark.py --profile full --seeds 20
```

LƯU Ý: full profile rất lớn. Nó được tạo để chạy benchmark thật, không phải smoke test.

## 6. Ablation

```powershell
python run_ablation.py --seeds 10
```

So sánh:

```text
graph_ga_full
graph_ga_no_articulation
graph_ga_no_connectivity_repair
graph_ga_no_guided
graph_ga_no_redundancy_mutation
graph_ga_no_redundancy_objective
vanilla_ga
```

## 7. Statistical test

Sau benchmark:

```powershell
python run_stats.py outputs/benchmark_quick/results.csv
```

Hoặc sau ablation:

```powershell
python run_stats.py outputs/ablation/results.csv --proposed graph_ga_full
```

## 8. Objective

Fitness scalar dùng cho Random / Greedy / GA / PSO / GraphAwareGA:

```text
fitness
= weighted_coverage
- redundancy_weight * redundancy_excess
- connectivity_weight * connectivity_deficit
- collision_weight * collision_ratio
```

NSGA-II không gộp các thành phần này ngay từ đầu. Nó tối ưu bốn mục tiêu riêng:

```text
maximize weighted coverage
minimize redundancy
minimize connectivity deficit
minimize collision ratio
```

Sau khi có Pareto front, code chọn nghiệm có scalar fitness tốt nhất để tiện benchmark
chung với các thuật toán khác.

## 9. GraphAwareGA hoạt động thế nào?

```text
population
   |
selection
   |
articulation-preserving crossover
   |
articulation-aware + coverage-guided mutation
   |
connectivity repair
   |
collision repair
   |
evaluate
```

### Articulation-aware mutation

Nếu UAV là articulation point của communication graph, mutation probability và step
size bị giảm.

Nếu UAV đang tạo nhiều sensing redundancy và không phải articulation point, nó được
phép mutation mạnh hơn.

### Coverage-guided mutation

Một số mutation không đi ngẫu nhiên mà hướng tới target chưa được phủ.

### Connectivity repair

Nếu graph bị tách thành nhiều component, thuật toán cố dịch chuyển nguyên một component
về phía component gần nhất. Dịch chuyển cả component giúp giữ cấu trúc liên kết nội bộ
tốt hơn việc kéo một UAV đơn lẻ.

### Collision repair

Các cặp UAV quá gần bị đẩy ra xa nhau, sau đó connectivity được repair lại.

## 10. Correctness tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Final solutions now expose `complete`, `in_bounds`, `feasible`, and
`constraint_violation` metrics. GraphAwareGA uses feasibility-first final output
selection while keeping the internal evolutionary selection based on the original
scalar fitness for a cleaner comparison.

## 11. Cấu trúc project

```text
checkpoint_02_graph_aware/
├─ CHECKPOINT.md
├─ README.md
├─ requirements.txt
├─ requirements-dev.txt
├─ tests/
├─ run_demo.py
├─ run_benchmark.py
├─ run_ablation.py
├─ run_stats.py
└─ src/
   ├─ problem.py
   ├─ objectives.py
   ├─ metrics.py
   ├─ graph_ops.py
   ├─ repair.py
   ├─ scenarios.py
   ├─ visualization.py
   ├─ baselines/
   │  ├─ random_search.py
   │  ├─ greedy.py
   │  ├─ vanilla_ga.py
   │  ├─ pso.py
   │  └─ nsga2.py
   ├─ proposed/
   │  └─ graph_aware_ga.py
   └─ experiments/
      ├─ configs.py
      ├─ benchmark.py
      ├─ ablation.py
      └─ statistics.py
```


## Checkpoint 03 — Formation transition

Nhánh này bắt đầu Checkpoint 03 nhưng vẫn giữ nguyên code Checkpoint 02.

Khác với giả định deployment từ một base cố định, CP3 định nghĩa bài toán tổng quát:

    formation A -> formation B

và tối ưu thời gian chuyển formation dưới các ràng buộc:

- giới hạn tốc độ;
- collision avoidance trong cả đoạn chuyển động, không chỉ ở endpoint;
- communication connectivity trong toàn bộ transition;
- map bounds.

Thiết kế và formulation đầy đủ nằm trong CHECKPOINT_03.md.

Chạy benchmark nhanh:

    python run_transition_benchmark.py --profile quick

Chạy connectivity-stress profile:

    python run_transition_benchmark.py --profile stress

Proposed planner hiện tại là BackboneTransitionPlanner. Mỗi transition segment bảo vệ một maximum-slack spanning tree, nhờ đó có certificate connectivity liên tục trong segment.


Joint formation + transition benchmark:

    python run_reconfiguration_benchmark.py --profile quick --seeds 3
    python run_reconfiguration_benchmark.py --profile stress --seeds 5

Paired statistics:

    python run_reconfiguration_stats.py outputs/reconfiguration_quick/results.csv

The main comparison is now:

    StaticThenTransition
        CP2 optimizes B without seeing deployment cost

    TransitionAwareGA
        CP2 graph-aware operators + exact A -> B transition cost in selection


### CP3 visualization

Interactive browser visualization:

    python run_visualize_reconfiguration.py --profile quick --problem-index 2 --algorithm both --budget quick --seed 0

Dense showcase:

    python run_visualize_reconfiguration.py --profile showcase --algorithm aware --budget standard --seed 0 --subframes 10 --frame-ms 60

The default output is a self-contained Plotly HTML plus a static PNG. The HTML
supports play/pause, a time slider, zoom, pan and hover. Add `--gif` only when a
GIF is useful for slides.

The final TransitionAwareGA prototype uses coverage-first hierarchical selection:

    feasible transition
    > weighted sensing coverage bucket
    > final CP2 static fitness bucket
    > shorter formation time
    > shorter total travel
    > exact sensing quality

This avoids sacrificing reachable sensing coverage merely to save a small amount
of transition time. The buckets deliberately let transition cost choose only
between near-tied formations. CP3 also disables CP2 articulation protection by
default so relay UAVs remain movable; connectivity is enforced by repair and the
transition planner.

Each frame shows:

- formation A and assigned formation B;
- current UAV positions and trajectory trails;
- current communication graph;
- protected backbone edges when available;
- rectangular no-fly obstacles;
- target points and currently covered targets;
- elapsed time, connectivity state and minimum UAV separation.

Outputs are written to:

    outputs/visualization/


### Obstacle-aware showcase

The CP3 showcase includes three static rectangular no-fly regions. The proposed
planner routes UAVs around them with a small visibility graph while continuing
to enforce communication connectivity and UAV-UAV separation.

    python run_visualize_reconfiguration.py --profile showcase --algorithm aware --budget standard --seed 0 --subframes 10 --frame-ms 60

Each command performs exactly one GA run. For the showcase visualization,
`standard` uses a larger population and refines several distinct finalists from
that same population to reduce seed sensitivity without restarting the solver.

Obstacle geometry is implemented with NumPy/basic geometry only; Plotly is used
only for the interactive browser visualization.


### Manual formation editor

Every interactive CP3 HTML now includes an **Edit B** mode.

1. Open the generated HTML.
2. Click **Edit B**.
3. Drag any numbered formation-B UAV node.
4. Coverage, smooth coverage potential, static fitness, communication connectivity, component count, minimum separation, collision count, obstacle feasibility, direct travel and straight-line time lower bound update live.
5. Click **Copy state** to copy the edited coordinates and metrics as JSON.

The editor is browser-side and intended for diagnosis/demo. It recomputes static formation metrics exactly, but it does not rerun the Python obstacle-aware transition planner. `directTravel` and `timeLB` are therefore straight-line estimates for the edited destination. Rerun the optimizer/planner when an exact transition trajectory is needed.

Entering Edit B hides the old trajectory/backbone because those paths belonged to the optimizer's original destination and would be misleading after manual editing.
