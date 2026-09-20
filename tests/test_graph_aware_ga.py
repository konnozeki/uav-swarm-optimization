from src.metrics import evaluate
from src.proposed import GraphAwareGA, GraphAwareConfig


def test_graph_aware_ga_returns_feasible_solution(simple_scenario):
    algorithm = GraphAwareGA(
        population_size=12,
        generations=8,
        elite_size=2,
    )

    solution, _ = algorithm.solve(simple_scenario, seed=7)
    metrics = evaluate(simple_scenario, solution)

    assert metrics.feasible


def test_config_switches_are_independent():
    config = GraphAwareConfig(
        use_articulation_awareness=False,
        use_redundancy_aware_mutation=True,
    )

    assert not config.use_articulation_awareness
    assert config.use_redundancy_aware_mutation
