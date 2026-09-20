# Checkpoint 03 — Connectivity-Preserving Formation Transition

## Research question

Checkpoint 02 tối ưu một formation tĩnh.

Checkpoint 03 hỏi:

> Cho hai formation hợp lệ A và B, làm thế nào để swarm chuyển từ A sang B nhanh nhất có thể trong khi vẫn tôn trọng speed limit, tránh collision và giữ communication connectivity trong suốt quá trình chuyển?

Điểm quan trọng: formation A không mặc định là một base.

Base deployment chỉ là một trường hợp đặc biệt:

    base/staging formation -> sensing formation

Cùng transition engine phải dùng được cho:

    formation A -> formation B

và sau này có thể mở rộng sang moving hotspot, online reconfiguration hoặc UAV failure recovery.

## Problem formulation

Input:

- formation A: X0 trong R^(N x 2)
- formation B: X* trong R^(N x 2)
- communication radius Rc
- minimum UAV separation d_min
- maximum speed V_max
- time step dt
- map bounds
- static rectangular no-fly obstacles
- obstacle clearance margin

Output:

    X0 -> X1 -> ... -> XT = X*

Constraints tại mọi transition step:

    ||x_i(t+1) - x_i(t)|| <= V_max * dt
    ||x_i(t) - x_j(t)|| >= d_min
    communication graph remains connected
    positions remain inside map bounds
    every UAV segment remains outside no-fly obstacles

Primary criterion:

    minimize formation-transition time

Secondary criterion:

    minimize total travel distance

Formation B hiện là input cố định. Joint optimization giữa chất lượng formation B và thời gian transition sẽ là bước mở rộng sau khi transition engine được kiểm chứng.

## Goal assignment

Nếu UAV được xem là đồng nhất, formation B là một tập slot không gắn cứng identity.

Checkpoint 03 dùng bottleneck assignment:

    min_pi max_i ||A_i - B_pi(i)||

để giảm lower bound của thời gian formation.

Sau khi tìm bottleneck threshold tối ưu, Hungarian assignment được dùng để minimize tổng distance trong tập assignment có cùng bottleneck optimum.

## Baselines

### DirectTransitionPlanner

Mỗi UAV bay thẳng tới goal với V_max.

Không kiểm tra connectivity hoặc collision.

Đây là lower-level time baseline và cho biết requirement an toàn làm transition chậm thêm bao nhiêu.

### EndpointGuardTransitionPlanner

Đề xuất direct step trước.

Nếu step làm mất connectivity hoặc gây collision, toàn bộ swarm step được backtrack.

Đây là reactive baseline: requirement chỉ được kiểm tra sau khi candidate movement đã được sinh.

## Proposed method: BackboneTransitionPlanner

Tại mỗi state, communication graph được chuyển thành một maximum-slack spanning tree.

Edge weight:

    slack(i,j) = Rc - distance(i,j)

Các edge ngắn hơn được ưu tiên vì có communication margin lớn hơn.

Trong transition tiếp theo, planner bảo vệ N-1 backbone edge này.

Nếu mỗi backbone edge thỏa:

    distance_start(i,j) <= Rc
    distance_end(i,j)   <= Rc

và UAV di chuyển tuyến tính giữa hai endpoint thì từ convexity của Euclidean norm:

    ||(1-tau) r_start + tau r_end||
    <= (1-tau)||r_start|| + tau||r_end||
    <= Rc

với mọi tau trong [0,1].

Do backbone là spanning tree, connectivity được certify trong toàn bộ segment, không chỉ ở discrete endpoints.

Backbone được rebuild ở state tiếp theo để swarm không bị khóa vào topology ban đầu.

## Continuous collision checking

Chỉ kiểm tra collision ở timestep đầu và cuối là không đủ vì hai UAV có thể cắt nhau ở giữa.

Với relative motion:

    r(tau) = r0 + tau * dr

Checkpoint 03 tìm minimum của ||r(tau)|| chính xác trên tau trong [0,1].

Do đó crossing trajectory được phát hiện ngay cả khi hai endpoint đều collision-free.

## Metrics

- formation time
- total travel distance
- max per-UAV travel distance
- speed feasibility
- discrete connected rate
- sampled continuous connected rate
- continuous minimum separation
- collision-free rate
- backbone certificate
- final feasibility
- straight-line time lower bound
- time efficiency

Time efficiency:

    T_lower_bound / T_actual

với:

    T_lower_bound = bottleneck_assignment_distance / V_max

## Run

Quick profile:

    python run_transition_benchmark.py --profile quick

Connectivity stress:

    python run_transition_benchmark.py --profile stress

Tests:

    python -m pytest -q

Outputs:

    outputs/transition_quick/
    outputs/transition_stress/
        results.csv
        summary.csv
        figures/

## Checkpoint boundary

Checkpoint 03 MVP cố ý giải bài toán:

    fixed formation A -> fixed formation B

Nó chưa jointly optimize sensing coverage và formation B.

Việc tách hai lớp là có chủ đích:

1. kiểm chứng connected/collision-free transition planning trước;
2. sau đó cho Checkpoint-02 formation optimization gọi transition planner trong fitness evaluation;
3. cuối cùng mới nghiên cứu Pareto trade-off giữa final coverage và deployment time.

Như vậy source của improvement vẫn được xác định rõ thay vì thay đổi static objective và transition mechanism cùng lúc.


## Joint formation + transition stage

Sau khi lower-level A -> B transition engine được tách riêng, branch này đã có
stage thứ hai:

    formation A -> optimize formation B -> deploy to B

Class chính:

    ReconfigurationProblem
    StaticThenTransition
    TransitionAwareGA

### StaticThenTransition

Baseline hai bước:

    GraphAwareGA(CP2) -> formation B
    BackboneTransitionPlanner(A, B)

Formation optimizer hoàn toàn không nhìn thấy transition time trong quá trình search.

### TransitionAwareGA

Chromosome vẫn chỉ chứa formation B:

    chromosome = [(x1,y1), ..., (xN,yN)]

Trajectory không được nhét vào chromosome. Mỗi candidate B được đánh giá bởi
BackboneTransitionPlanner.

Final prototype selection dùng thứ tự ưu tiên thay vì ép mọi mục tiêu vào
một weighted sum duy nhất:

    1. feasible A -> B transition
    2. maximize weighted sensing coverage
    3. maximize final CP2 static fitness
    4. minimize normalized formation time
    5. minimize normalized total travel

Nói cách khác, transition cost chỉ được dùng để phân biệt các formation có sensing
quality tương đương. Một UAV không được phép đứng gần formation A chỉ để tiết kiệm
thời gian nếu việc di chuyển nó có thể tăng coverage.

Code vẫn tính diagnostic joint_fitness:

    joint_fitness
      = checkpoint_02_static_fitness
      - lambda_t * normalized_transition_time
      - lambda_d * normalized_travel
      - infeasible_transition_penalty

với:

    normalized_transition_time
      = T_form / (map_diagonal / V_max)

    normalized_travel
      = total_travel / (N * map_diagonal)

Mặc định diagnostic weights vẫn là:

    lambda_t = 0.20
    lambda_d = 0.05

Weighted-sum selection cũ vẫn có thể chạy bằng selection_mode="weighted_joint" để
reproduce/ablate thiết kế trước đó.

CP3 cũng mặc định tắt articulation-aware protection của CP2. Ablation CP2 không cho
thấy đóng góp có ý nghĩa thống kê của cơ chế này, và trong reconfiguration nó có thể
khóa chính các relay UAV cần dịch chuyển để kéo cả connected formation tới vùng
target chưa được phủ.

TransitionAwareGA cũng giữ formation A như một anchor candidate với transition cost
bằng 0, đồng thời warm-start một phần population quanh A. Phần population còn lại
vẫn được khởi tạo toàn cục để tránh ép search chỉ quanh formation hiện tại.

### Joint benchmark

Quick:

    python run_reconfiguration_benchmark.py --profile quick --seeds 3

Stress:

    python run_reconfiguration_benchmark.py --profile stress --seeds 5

Statistics:

    python run_reconfiguration_stats.py outputs/reconfiguration_quick/results.csv

Kết quả gồm:

- final weighted coverage
- final CP2 fitness
- joint fitness
- transition feasibility
- formation time
- total travel distance
- time efficiency
- runtime
- paired Wilcoxon + Holm correction
- paired feasibility counts

Checkpoint-03 hypothesis cho stage này:

H1. Transition-aware selection giảm formation time so với static-then-transition
trong khi giữ final coverage gần tương đương.

H2. Transition-aware selection giảm tỷ lệ destination formation không thể đạt được
bằng connected/collision-free transition.

H3. Lợi ích tăng khi communication radius giảm hoặc current formation A ở xa vùng
sensing tốt.



## Reconfiguration ablation

Run:

    python run_reconfiguration_ablation.py --seeds 5

Variants:

    transition_aware_full
    transition_aware_no_warm
    transition_aware_weighted_joint
    static_then_transition

Interpretation:

- full vs no_warm isolates temporal population initialization;
- full vs weighted_joint isolates hierarchical coverage-first selection from the older scalar trade-off;
- full vs static_then_transition measures the complete CP3 contribution.

Trajectory figures are also written for the representative seed so deadlock and
topology behavior can be inspected visually instead of relying only on aggregate
CSV metrics.


## Visualization

Checkpoint 03 includes a lightweight visualization layer because transition
quality is difficult to interpret from CSV metrics alone.

Run:

    python run_visualize_reconfiguration.py --profile quick --problem-index 2 --algorithm both --budget quick --seed 0

For each selected algorithm the command writes:

    <case>__<algorithm>__seed<seed>.png
    <case>__<algorithm>__seed<seed>.gif

The visualization uses basic Matplotlib primitives only. It is not part of the
optimization objective and does not affect correctness metrics.

Persistent marks:

- hollow circles: formation A;
- x markers: assigned slots in formation B;
- filled circles: current UAV positions;
- thin lines: current communication graph;
- stronger lines: protected backbone for the active segment;
- trails: UAV motion history;
- target dots: sensing targets, with currently covered targets emphasized.

Frame text reports elapsed time, connectivity, minimum pairwise separation and
instantaneous binary coverage. Research coverage/fitness values continue to come
from the normal metrics pipeline.


## Obstacle-aware transition

Checkpoint 03 now models static axis-aligned rectangular no-fly regions.

The UAV itself remains a point in the 2D model. A configurable
obstacle_clearance expands each rectangle before geometric checks, so the
planner still keeps a physical safety margin.

For every transition segment the evaluator checks exact line-segment versus
rectangle intersection. Therefore a trajectory is invalid even when both
segment endpoints are outside the obstacle but the line between them crosses
the blocked region.

BackboneTransitionPlanner uses a small visibility graph for detours:

    current UAV position
        + assigned goal
        + expanded obstacle corners
        -> shortest visible polygonal route
        -> first waypoint
        -> bounded motion step
        -> backbone projection
        -> continuous obstacle/collision/connectivity checks

The visibility graph is rebuilt online, so a UAV returns to moving directly
toward its goal as soon as the goal becomes visible.

Important modeling boundary:

- obstacles block UAV motion;
- they do not currently block radio communication or sensing line-of-sight.

Those effects can be introduced later as separate propagation/sensing models
without changing the geometric no-fly constraint.

The dense showcase profile contains three rectangular obstacles and is intended
for visual inspection:

    python run_visualize_reconfiguration.py --profile showcase --algorithm aware --budget standard --seed 0 --subframes 10 --frame-ms 60


## Connected-group coverage mutation

Binary sensing coverage creates a search plateau: moving closer to an uncovered
target gives no reward until the sensing radius is actually crossed. Moving one
relay UAV can also break communication and be undone by connectivity repair.

The final CP3 prototype therefore adds two search-only mechanisms without
changing the reported binary coverage metric.

Smooth coverage potential:

    gap_m = max(min_i distance(target_m, UAV_i) - Rs, 0)

    potential_m = exp(-gap_m / sigma)

Covered targets have potential 1. An uncovered target gradually approaches 1 as
the swarm moves toward it.

Smooth coverage potential is a search signal only. It is used to keep useful
"almost there" offspring alive and inside temporary local probe chains, but it
is excluded from final solution ordering. Therefore the returned formation does
not pay extra transition cost merely to stand closer to a target it still does
not cover.

Connected-group mutation:

1. sample a cluster of currently uncovered targets;
2. choose candidate anchors from UAVs near that cluster and UAVs with low unique
   sensing contribution;
3. take graph neighborhoods around those anchors;
4. translate the whole connected group toward the cluster;
5. preserve active boundary communication edges while moving;
6. repair collision/connectivity/obstacle endpoint constraints;
7. accept the mutation only when binary coverage, or on a tie the smooth
   potential, improves.

This specifically lets relay chains move together instead of repeatedly moving
one UAV and having connectivity repair pull it back.


## Final local cleanup

After the evolutionary loop chooses its best candidate, CP3 performs two small
deterministic cleanup stages on that one candidate only.

Coverage probe:

- repeatedly apply forced connected-group moves through a temporary probe chain;
- potential-only intermediate moves are allowed inside the probe;
- commit the probe only if the real final ordering improves, normally by
  increasing binary coverage.

Motion pruning:

- use the current UAV-to-goal assignment;
- try retracting long destination slots toward the corresponding start UAV;
- re-evaluate the complete A -> B transition;
- keep a retraction only when coverage/static quality are preserved and the
  final ordering improves.

The lower-level BackboneTransitionPlanner also shortcuts its completed
trajectory. It replaces waypoint zig-zags by a straight speed-limited
interpolation only when the shortcut is collision-free, obstacle-free and has a
spanning-tree connectivity certificate at every new segment. This removes
unnecessary intermediate motion without weakening transition constraints.


## Manual diagnostic editor

The interactive Plotly artifact supports a browser-side Edit B mode for diagnosing local optimizer failures.

Formation-B UAV nodes can be dragged directly. The browser immediately recomputes the static destination metrics:

- weighted sensing coverage;
- smooth uncovered-target potential;
- CP2 scalar static fitness;
- communication graph connectivity and component count;
- minimum UAV separation and collision violations;
- obstacle endpoint feasibility;
- total straight-line displacement from A;
- straight-line formation-time lower bound.

Communication edges and covered-target highlighting are redrawn live.

The last two motion values are diagnostic lower bounds only. Manual edits do not rerun BackboneTransitionPlanner in JavaScript, so exact obstacle-aware trajectory feasibility/time still requires a Python planner run.

Copy state exports all edited coordinates and the live metrics as JSON, making visual counterexamples easy to reproduce or discuss.


## Deterministic coverage-first refinement

Manual drag counterexamples showed that the stochastic GA could stop at a reachable but clearly under-covered formation even when a coordinated relay move produced much higher coverage.

The final prototype therefore applies a deterministic refinement after GA search:

1. find the major clusters of still-uncovered targets;
2. choose nearby and low-contribution UAV anchors;
3. generate elastic communication-neighborhood pulls toward each cluster;
4. cheaply pre-screen candidates with static coverage, smooth potential and static fitness;
5. run the exact obstacle-aware A -> B planner only for the strongest candidates;
6. accept improvements using the final lexicographic order:

    feasible transition
    > binary weighted coverage
    > static CP2 fitness
    > lower formation time
    > lower total travel

Transition cost therefore cannot block a real coverage increase. Smooth potential is used only to traverse temporary same-coverage plateaus during search/refinement.

The visualizer exposes `--coverage-refine-rounds` and `--coverage-refine-exact-candidates` for diagnosis. Defaults are 6 rounds and 12 exact candidates per round.
