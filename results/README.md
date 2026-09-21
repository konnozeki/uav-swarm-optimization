# Final experiment evidence

This directory is reserved for frozen result artifacts used by the final analysis/report.
Exploratory runs should continue to use `outputs/`.

Expected CP3 layout after the final campaign:

    results/cp3_final/
        benchmark/
            experiment_manifest.json
            results.csv
            summary.csv
            statistics.csv
            feasibility_comparison.csv
        ablation/
            experiment_manifest.json
            results.csv
            summary.csv
        sensitivity/
            experiment_manifest.json
            results.csv
            summary.csv

Do not manually edit generated CSV files. Regenerate them from the recorded commit and manifest.