from pathlib import Path

from src.scenarios import make_scenario
from src.baselines import VanillaGA
from src.proposed import GraphAwareGA
from src.metrics import evaluate
from src.visualization import plot_solution


def main():
    scenario = make_scenario(
        pattern="split",
        seed=42,
        n_uavs=7,
        n_targets=100,
        communication_radius=250,
    )

    algorithms = [
        VanillaGA(
            population_size=50,
            generations=70,
        ),
        GraphAwareGA(
            population_size=50,
            generations=70,
        ),
    ]

    out = Path("outputs/demo")
    out.mkdir(parents=True, exist_ok=True)

    for algorithm in algorithms:
        solution, runtime = algorithm.solve(
            scenario,
            seed=42,
        )
        metrics = evaluate(scenario, solution)

        print()
        print(f"[{algorithm.name}]")
        print(f"coverage              = {metrics.weighted_coverage_ratio:.4f}")
        print(f"redundancy            = {metrics.redundancy_excess:.4f}")
        print(f"connected             = {metrics.connected}")
        print(f"connectivity_deficit  = {metrics.connectivity_deficit:.4f}")
        print(f"collision_ratio       = {metrics.collision_ratio:.4f}")
        print(f"complete              = {metrics.complete}")
        print(f"in_bounds             = {metrics.in_bounds}")
        print(f"feasible              = {metrics.feasible}")
        print(f"constraint_violation  = {metrics.constraint_violation:.4f}")
        print(f"fitness               = {metrics.fitness:.4f}")
        print(f"runtime               = {runtime:.3f}s")

        plot_solution(
            scenario,
            solution,
            output_path=out / f"{algorithm.name}.png",
        )


if __name__ == "__main__":
    main()
