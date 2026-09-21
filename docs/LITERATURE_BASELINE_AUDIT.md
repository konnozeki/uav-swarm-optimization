# Literature baseline fidelity audit

This document records what can and cannot be claimed about the recent UAV-domain
baselines used by CP3.

## Public source-code search

Searches were performed by exact paper title, method acronym, author names and DOI
on GitHub and the public web.

### JOCC CPGS / DPGS

Paper:

Lei Wang, Tingting Fan, Ping Wang, Linfeng Liu,
"Sustaining connectivity and expanding coverage: UAV swarm deployment strategies
for joint connectivity-coverage optimization", Computer Networks 287 (2026) 112565.
DOI: 10.1016/j.comnet.2026.112565.

No official/public repository from the authors was found in the searches performed
for this audit. The article states that evaluation used a Python simulator, but the
public article metadata does not expose a source-code repository.

Therefore the repository implementation MUST be described as an adaptation, not an
author-code reproduction.

What is supported directly by the paper and retained here:

- joint connectivity/coverage deployment;
- hard binary graph for connectivity measurement;
- differentiable soft graph for gradient optimization;
- algebraic connectivity as the connectivity signal;
- responsibility-area coverage decomposition;
- centralized projected-gradient strategy using global state;
- distributed projected-gradient strategy using local/neighbor information.

What is adaptation-specific in this repository:

- 2D weighted target points replace the paper's continuous ground-area / camera
  coverage formulation;
- the Gaussian target responsibility kernel is an implementation choice;
- the Gaussian soft communication edge weight is an implementation choice;
- minimum-separation handling and `repair_solution` are CP3 constraints/projection,
  not a recovered author implementation;
- initialization and stopping/budget are CP3 choices.

Conclusion: `jocc_cpgs_2026` and `jocc_dpgs_2026` are mechanism-faithful
literature adapters, not reproduction-grade implementations.

## R2C

Paper:

Yabin Peng et al.,
"Resilient UAV Swarm with Fast Connectivity Recovery and Extensive Coverage",
AAAI 2026, 917-925.
DOI: 10.1609/aaai.v40i2.37060.

No official/public repository from the authors was found in the searches performed
for this audit. The AAAI article page exposes the PDF/poster but no code link.

The paper specifies the ISE equations explicitly. The implementation
`R2CBufferedVirtualForce` now follows the published component equations:

- Eq. (9): buffered pairwise force with repulsive / neutral / attractive zones;
- Eq. (10): inward edge force near deployment boundaries;
- Eq. (11): total ISE force is neighbor force sum plus boundary force;
- Eq. (12): velocity follows the normalized total-force direction and is capped by
  a configured maximum speed.

The full R2C method is NOT reproduced. Full R2C additionally requires:

- failure-induced partition into subnetworks;
- convex-hull boundary-node extraction and virtual multipartite graph construction;
- learned multipartite graph convolution, including trained weights;
- subnetwork-level translation action;
- connectivity/time/movement training losses;
- adaptive time-aware fusion of ISE and IST actions.

Those learned artifacts/training code were not publicly located, so calling the
current implementation "R2C" would overstate fidelity. It remains deliberately
named `r2c_ise_aaai26`.

For the common CP3 benchmark, the paper's physical maximum speed is not part of the
static `Scenario` interface. The adapter therefore keeps the historical
scale-relative speed fallback unless `R2CBufferedForceConfig.max_speed` is
explicitly supplied. A paper-validation runner should set the physical value
directly.

## Claims allowed in the report

Safe wording:

> Recent UAV-domain baselines were independently implemented from the published
> algorithms. JOCC CPGS/DPGS are adapted to the common weighted target-point model.
> For R2C, the published ISE force equations are implemented directly, while the
> learned IST/AWF modules are not reproduced because no public author code or
> trained artifacts were located.

Avoid:

> We reproduced the authors' code.

and avoid:

> Full R2C was implemented.

The common CP3 benchmark answers whether these adapted mechanisms perform well under
our unified problem definition. It does not reproduce the papers' original reported
scores.
