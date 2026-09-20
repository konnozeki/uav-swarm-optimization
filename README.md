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
