# Domain / literature baselines

The final CP3 benchmark includes recent UAV-specific methods in addition to generic
optimizers. These are adaptations to the repository's common 2D weighted-target
coverage model, not claims of bit-for-bit reproduction when the source paper solves
a materially different sensing/control problem.

## 1. JOCC CPGS adapter

Source: Lei Wang, Tingting Fan, Ping Wang, Linfeng Liu,
"Sustaining connectivity and expanding coverage: UAV swarm deployment strategies
for joint connectivity-coverage optimization," Computer Networks, vol. 287,
article 112565, September 2026.
DOI: 10.1016/j.comnet.2026.112565

The paper is the closest recent match to CP2/CP3's static destination problem:
it explicitly formulates joint UAV-swarm connectivity and coverage and proposes a
Centralized Projected Gradient Strategy (CPGS). Its dual-graph design uses a hard
binary connectivity graph together with a differentiable soft graph, while coverage
is handled through responsibility-area partitioning and global projected-gradient
updates.

Repository implementation: `JOCCCentralizedProjectedGradient`.

Adaptation boundary:
- 2D target points replace the paper's continuous ground-area/camera coverage model;
- soft target responsibilities act as the discrete responsibility partition;
- a Gaussian weighted graph supplies the differentiable soft communication graph;
- the Fiedler-vector derivative supplies algebraic-connectivity ascent;
- `repair_solution` is the hard projection used to enforce the repository's
  connectivity and minimum-separation constraints.

## 2. JOCC DPGS adapter

Same source and DOI as CPGS.

The paper's Distributed Projected Gradient Strategy (DPGS) lets each UAV update
using local information, targeting lower complexity and communication overhead than
the centralized method.

Repository implementation: `JOCCDistributedProjectedGradient`.

Adaptation boundary:
- each UAV assigns target responsibility against itself and its one-hop neighbors;
- cohesion/separation forces use only one-hop communication neighbors;
- all UAVs apply synchronous local updates before the common hard feasibility
  projection.

## 3. R2C buffered virtual-force component

Source: Yabin Peng, Chenyu Zhou, Hainan Cui, Tong Duan, Haoyang Chen, Fan Zhang,
Shaoxun Liu, "Resilient UAV Swarm with Fast Connectivity Recovery and Extensive
Coverage," AAAI 2026, pp. 917-925.
DOI: 10.1609/aaai.v40i2.37060

R2C is aimed at self-healing after UAV failures rather than destination-formation
optimization. Its first stage, intra-subnetwork expansion (ISE), is nevertheless a
directly relevant recent domain heuristic for balancing connectivity and spatial
coverage. The paper uses a buffered spring with repulsive, neutral and attractive
zones and reports `[d_l, d_u] = [0.8 R_c, 0.9 R_c]`.

Repository implementation: `R2CBufferedVirtualForce`.

Important: this is deliberately named `r2c_ise_aaai26`, not `R2C`. The full paper
also contains a trained multipartite graph-convolution translation module and
adaptive fusion for disconnected subnetworks. Reproducing that full model would
require the original failure-recovery training setting and would not be a fair
drop-in static target-coverage baseline.

## CP3 comparison protocol

Each static literature method chooses destination formation B without seeing A->B
transition cost. `LiteratureStaticThenTransition` then applies the exact same
`BackboneTransitionPlanner` used by the existing `StaticThenTransition` baseline.
This isolates the quality of the destination optimizer. `TransitionAwareGA` remains
the only final benchmark method that places exact transition evaluation inside the
destination search.

The default final benchmark therefore contains:

    static_then_transition
    jocc_cpgs_2026_then_transition
    jocc_dpgs_2026_then_transition
    r2c_ise_aaai26_then_transition
    transition_aware_ga

Statistics are paired on the same `(problem, seed)` and Holm correction is applied
across all baseline/metric Wilcoxon tests. Transition feasibility uses exact
McNemar tests with a separate Holm correction family.
