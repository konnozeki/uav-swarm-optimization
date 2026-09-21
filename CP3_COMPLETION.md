# CP3 completion gate

Checkpoint 04 should not start until the CP3 research package below is frozen.

- [ ] Final CP3 algorithm frozen
- [x] Final mechanism flags exposed for ablation
- [x] One-component-off CP3 ablation framework
- [x] Dedicated hyperparameter sensitivity framework
- [x] Final 24-seed experiment runner
- [x] Experiment manifest generation
- [x] CP3 complexity accounting scaffold
- [ ] Final benchmark campaign completed
- [ ] CP3 ablation campaign completed
- [ ] Sensitivity campaign completed
- [ ] 20-30 seed final evidence stored
- [ ] Statistical tests generated from final evidence
- [x] Domain/literature baselines selected and implemented (JOCC CPGS, JOCC DPGS, R2C-ISE adapter)
- [ ] Public/external benchmark integrated
- [ ] Reproducibility environment pinned
- [ ] Final quantitative case study frozen
- [ ] Full test suite passing on the frozen commit
  - Current `cp3-completion` CI passes, but this item remains open until the final frozen commit.
- [ ] checkpoint-03-final tag/release created

## Final evidence command

Default final evidence uses 24 paired seeds:

    python run_cp3_final.py --phase benchmark
    python run_cp3_final.py --phase ablation
    python run_cp3_final.py --phase sensitivity
    python run_cp3_final.py --phase statistics

`--phase all` is intentionally available but can be expensive.

Exploratory outputs remain under `outputs/`. Final evidence belongs under `results/cp3_final/`.