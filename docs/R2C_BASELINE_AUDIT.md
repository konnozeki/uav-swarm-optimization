# R2C-ISE baseline audit

## Scope

The repository baseline named `r2c_ise_aaai26` implements only the
intra-subnetwork expansion (ISE) component from:

Y. Peng et al., "Resilient UAV Swarm with Fast Connectivity Recovery and
Extensive Coverage," AAAI 2026, pp. 917--925,
DOI `10.1609/aaai.v40i2.37060`.

It is not a reproduction of full R2C. In particular, it does not implement the
trained inter-subnetwork graph-convolution module or adaptive fusion mechanism.

## Equation mapping

- Equation (9): `_pair_force` implements the repulsive, neutral and attractive
  distance zones. The communication radius comes from the current scenario.
- Equation (10): `_boundary_force` adds exponentially decaying inward forces
  near each map boundary.
- Equation (11): `_force` sums pairwise and boundary forces.
- Equation (12): `_velocity` converts total force into a velocity capped by
  `max_speed`.

The paper reports `[d_l, d_u] = [0.8 R_c, 0.9 R_c]`, decay coefficients of
`0.01`, maximum speed `1 m/s`, and a horizon of `400` time steps. These are the
adapter defaults.

## Repository-specific adaptations

- The paper does not publish a numerical boundary-buffer width `d_e`. The
  adapter exposes `boundary_buffer_ratio`, defaulting to 10% of the shorter map
  side. This is an explicit project assumption, not a paper value.
- CP3 optimizes weighted target points rather than continuous communication
  coverage after failures. The adapter initializes a connected formation near
  the weighted target centroid before applying ISE.
- `repair_solution` projects each update back into the repository's hard
  connectivity and collision constraints. This projection is not part of R2C.
- CP3 uses the scenario's communication radius rather than forcing the paper's
  experimental value of 120 m.

Results must therefore be described as an **R2C-ISE component adapter**, not as
full R2C or a bit-for-bit reproduction.

## Regression coverage

`tests/test_domain_baselines.py` checks numerical values for all three zones in
equation (9), the direction and magnitude of equation (10), and the velocity
formula and speed cap in equation (12). The existing smoke test continues to
check that the adapted solver returns a complete feasible solution.
