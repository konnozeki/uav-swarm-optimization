# Checkpoint 02 — Graph-aware optimization

## Storyline

Checkpoint 01 trả lời:

> Ta có thể mô hình hóa bài toán, tính coverage/connectivity/collision/redundancy,
> và chạy các baseline cơ bản hay không?

Checkpoint 02 trả lời:

> Nếu evolutionary algorithm hiểu cấu trúc graph thay vì chỉ bị phạt sau khi
> graph bị đứt, chất lượng nghiệm có cải thiện không?

## Những gì mới so với Checkpoint 01

1. Objective được tách thành các thành phần rõ ràng:
   - weighted coverage
   - redundancy
   - connectivity deficit
   - collision ratio

2. Baseline mạnh hơn:
   - Random Search
   - Connected Greedy
   - Vanilla GA
   - Particle Swarm Optimization
   - NSGA-II

3. Thuật toán đề xuất:
   - GraphAwareGA
   - articulation-aware mutation
   - articulation-preserving crossover
   - connectivity repair
   - collision repair
   - coverage-guided mutation

4. Benchmark:
   - quick profile
   - connectivity-stress profile
   - full parameter sweep generator

5. Ablation:
   - full GraphAwareGA
   - no articulation awareness
   - no connectivity repair
   - no coverage-guided mutation
   - no redundancy term
   - vanilla GA

6. Statistical test:
   - paired Wilcoxon test
   - Holm correction
   - paired median difference

## Research hypothesis

H1:
Graph-aware evolutionary optimization cải thiện coverage/connectivity trade-off
so với evolutionary optimization chỉ dùng penalty.

H2:
Connectivity repair đem lại phần lớn lợi ích trong các scenario mà communication
radius trở thành bottleneck.

H3:
Articulation-aware mutation giảm số lần evolutionary search phá các UAV đang giữ vai trò
cầu nối mà không làm tăng bậc complexity chính quá nhiều.

## Complexity notation

Gọi:
- N = số UAV
- M = số target
- E_eval = O(MN + N^2), chi phí một lần đánh giá nghiệm
- P = population size
- G = số generation
- C = số candidate của greedy
- S = số random samples

Xấp xỉ:

- Random Search: O(S * E_eval)
- Greedy: O(N * C * E_eval)
- Vanilla GA: O(G * P * E_eval)
- PSO: O(G * P * E_eval)
- NSGA-II: O(G * (P * E_eval + P^2 * D)), D là số objective
- GraphAwareGA:
  O(G * P * (E_eval + N^2 + R(N)))

Trong đó graph articulation/connectivity check tối đa O(N^2) trên graph dày,
và R(N) là chi phí repair. Với swarm nhỏ/vừa, phần MN của coverage evaluation
thường vẫn là thành phần lớn.

Checkpoint 02 chưa tuyên bố đây là thuật toán cuối cùng. Mục đích của checkpoint
là tạo ra một mechanism đủ rõ để benchmark + ablation và từ đó quyết định hướng
cho checkpoint tiếp theo.
