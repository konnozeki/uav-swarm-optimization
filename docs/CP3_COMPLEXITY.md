# Checkpoint 03 complexity analysis

This document freezes the notation and accounting used for the final CP3 report.

## Notation

- N: number of UAVs
- M: number of sensing targets
- P: evolutionary population size
- G: number of generations
- R: number of deterministic refinement rounds
- C: exact refinement candidates retained per round
- H: maximum graph-hop radius used by connected-group search
- T: number of transition steps produced by the lower-level planner
- O: number of rectangular no-fly obstacles

## Static formation evaluation

Coverage checks all target/UAV pairs: O(MN).
Dense communication-graph construction and collision checks are O(N^2).

Therefore:

    E_static = O(MN + N^2)

## BackboneTransitionPlanner

For one transition step:

1. communication graph construction: O(N^2)
2. maximum spanning tree on a dense graph: O(N^2 log N)
3. bounded backbone projection: O(I_proj N)
4. continuous pairwise collision check: O(N^2)
5. obstacle segment checks: O(NO)
6. local visibility routing depends polynomially on the obstacle visibility graph

Ignoring obstacle-visibility constants:

    E_transition = O(T (N^2 log N + N^2 + NO))

Trajectory shortcutting is reported separately because its effective cost depends on shortcut_lookahead.

## TransitionAwareGA

Each exact candidate requires static evaluation plus a full A -> B transition evaluation.

    O(GP (E_static + E_transition))

Deterministic refinement adds at most R rounds and C exact candidates per round:

    O(RC (E_static + E_transition))

Thus the upper-level CP3 accounting is:

    O((GP + RC) (E_static + E_transition))

plus lower-order evolutionary operators and final motion pruning.

## Required empirical scaling

The final evidence must report runtime versus N, M, obstacle count, P and G.
Where practical, profile the share spent in exact transition evaluation/refinement.
These measurements are the bridge to the CP4 realtime design.