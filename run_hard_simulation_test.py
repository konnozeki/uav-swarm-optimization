"""Script to simulate UAV Swarm Reconfiguration in a Hard/Stressful Scenario.

This test validates whether the UAVs actually reach the target solution (goal endpoints)
step-by-step under strict kinematic, collision-free, connectivity, and obstacle constraints.
"""

from __future__ import annotations

import time
import numpy as np

from src.problem import Scenario, DEFAULT_MIN_SEPARATION
from src.reconfiguration import ReconfigurationProblem
from src.obstacles import AxisAlignedRectangle
from src.proposed.hierarchical_reconfiguration import HierarchicalReconfiguration
from src.transition import (
    TransitionProblem,
    evaluate_transition,
)
from src.plotly_transition_visualization import save_transition_html


def create_hard_stress_scenario() -> ReconfigurationProblem:
    """Create a difficult simulation case with:
    1. Narrow obstacle wall separating Start and Target area.
    2. Tight communication radius forcing swarm connectivity maintenance.
    3. Strict min separation to test inter-UAV collision avoidance.
    4. Target cluster placement requiring formation transformation (e.g. Ring -> Narrow Line -> Target).
    """
    map_w, map_h = 1000.0, 1000.0
    n_uavs = 8
    
    # 1. Target clusters concentrated at top-right area (x ~ 800, y ~ 800)
    np.random.seed(42)
    t_cluster1 = np.random.normal(loc=[800.0, 800.0], scale=[60.0, 60.0], size=(60, 2))
    t_cluster2 = np.random.normal(loc=[850.0, 650.0], scale=[50.0, 50.0], size=(40, 2))
    targets = np.clip(np.vstack([t_cluster1, t_cluster2]), 0, 1000.0)
    
    n_targets = len(targets)
    scenario = Scenario(
        name="hard_narrow_corridor_stress",
        pattern="custom_stress",
        width=map_w,
        height=map_h,
        targets=targets,
        target_weights=np.ones(n_targets),
        communication_radius=220.0,  # Tight connectivity constraint for 8 UAVs spanning long distances
        sensing_radius=150.0,
        min_separation=DEFAULT_MIN_SEPARATION,
        n_uavs=n_uavs,
    )
    
    # 2. Start positions: Swarm in Ring formation at bottom-left (x ~ 200, y ~ 200)
    center_start = np.array([200.0, 200.0])
    angles = np.linspace(0, 2 * np.pi, n_uavs, endpoint=False)
    r_start = 80.0
    start_positions = center_start + r_start * np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 3. No-Fly Obstacles: Central barrier with a corridor gap between (420, 350) and (580, 650)
    obstacles = (
        # Left barrier block
        AxisAlignedRectangle(x_min=0.0, y_min=420.0, x_max=420.0, y_max=580.0, name="obstacle_left"),
        # Right barrier block
        AxisAlignedRectangle(x_min=580.0, y_min=420.0, x_max=1000.0, y_max=580.0, name="obstacle_right"),
    )
    
    return ReconfigurationProblem(
        name="hard_corridor_reconfig",
        scenario=scenario,
        start_positions=start_positions,
        max_speed=30.0,
        dt=1.0,
        allow_reassignment=True,
        obstacles=obstacles,
        obstacle_clearance=10.0,
    )


def simulate_and_verify() -> None:
    print("=" * 80)
    print("STARTING UAV SWARM HARD CASE RECONFIGURATION SIMULATION & VERIFICATION")
    print("=" * 80)
    
    problem = create_hard_stress_scenario()
    print(f"Problem: {problem.name}")
    print(f"UAV count: {problem.scenario.n_uavs}")
    print(f"Communication Radius (Rc): {problem.scenario.communication_radius}")
    print(f"Min Separation (d_min): {problem.scenario.min_separation}")
    print(f"Max Speed: {problem.max_speed} m/s | dt: {problem.dt} s")
    print(f"Obstacles count: {len(problem.obstacles)}")
    print("-" * 80)
    
    # Use TransitionAwareGA for joint sensing + transition optimization
    from src.proposed.transition_aware_ga import TransitionAwareGA
    
    solver = TransitionAwareGA(
        population_size=12,
        generations=15,
        selection_mode="coverage_first",
    )
    
    start_time = time.time()
    solution, solve_runtime = solver.solve(problem, seed=42)
    elapsed = time.time() - start_time
    
    print(f"Solver completed in {elapsed:.2f} seconds.")
    print(f"Algorithm Name: {solution.algorithm}")
    print(f"Is Joint Solution Feasible: {solution.evaluation.feasible}")
    print(f"Weighted Coverage Ratio achieved: {solution.evaluation.weighted_coverage_ratio:.4f}")
    print(f"Final Fitness: {solution.evaluation.final_fitness:.4f}")
    
    # Extract trajectory
    route = solution.transition_solution
    if route is None or not route.reached_goal:
        print("[FAIL] Solver failed to generate a feasible trajectory!")
        return

    trajectory = route.trajectory  # Shape: (T, N, 2)
    n_steps, n_uavs, _ = trajectory.shape
    print(f"Generated Trajectory Duration: {n_steps} timesteps ({n_steps * problem.dt:.1f} seconds).")
    
    # --- Step-by-Step Simulation Check ---
    print("\n--- STEP-BY-STEP KINEMATIC & CONSTRAINT SIMULATION ---")
    
    start_pos = trajectory[0]
    final_pos = trajectory[-1]
    assigned_goals = route.assigned_goals
    
    # 1. Start Position Check
    start_dist_err = np.linalg.norm(start_pos - problem.start_positions)
    print(f"[Check 1] Trajectory Start matches Initial Swarm State: Error = {start_dist_err:.6f} m")
    
    # 2. Final Position Reachability Check
    end_dist_err = np.linalg.norm(final_pos - assigned_goals)
    max_individual_goal_err = np.max(np.linalg.norm(final_pos - assigned_goals, axis=1))
    print(f"[Check 2] Trajectory Endpoint matches Computed Target Solution:")
    print(f"          Max individual UAV error to goal endpoint: {max_individual_goal_err:.6f} m")
    
    # 3. Kinematic Max Speed Check
    max_step_speed = 0.0
    for t in range(1, n_steps):
        step_disp = np.linalg.norm(trajectory[t] - trajectory[t - 1], axis=1) / problem.dt
        max_step_speed = max(max_step_speed, np.max(step_disp))
    print(f"[Check 3] Max observed UAV speed across trajectory: {max_step_speed:.4f} m/s (Limit: {problem.max_speed} m/s)")
    
    # 4. Inter-UAV Min Separation Collision Check
    min_observed_sep = float("inf")
    for t in range(n_steps):
        pos_t = trajectory[t]
        diff = pos_t[:, None, :] - pos_t[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_observed_sep = min(min_observed_sep, np.min(dist_matrix))
    print(f"[Check 4] Minimum inter-UAV separation during motion: {min_observed_sep:.4f} m (Required >= {problem.scenario.min_separation} m)")
    
    # 5. Obstacle Avoidance Verification
    trans_problem = TransitionProblem(
        name="verify_trans",
        width=problem.scenario.width,
        height=problem.scenario.height,
        start_positions=problem.start_positions,
        goal_positions=assigned_goals,
        communication_radius=problem.scenario.communication_radius,
        min_separation=problem.scenario.min_separation,
        max_speed=problem.max_speed,
        dt=problem.dt,
        allow_reassignment=False,
        obstacles=problem.obstacles,
        obstacle_clearance=problem.obstacle_clearance,
    )
    
    val_result = evaluate_transition(trans_problem, route)
    print(f"[Check 5] Full Obstacle & Bounds Clearance Validation: {val_result.feasible}")
    if not val_result.feasible:
        print(f"          Obstacle free: {val_result.obstacle_free}, Collision free: {val_result.collision_free}, In bounds: {val_result.in_bounds}")
    
    # Save interactive HTML visualization
    html_out = "outputs/hard_simulation_test.html"
    save_transition_html(
        problem=trans_problem,
        solution=route,
        output_path=html_out,
        title=f"Hard Case Simulation Test - {problem.name}",
        targets=problem.scenario.targets,
        sensing_radius=problem.scenario.sensing_radius,
        target_weights=problem.scenario.target_weights,
    )
    print(f"\n[HTML Render] Saved interactive animation to: {html_out}")

    # Summary
    print("\n" + "=" * 80)
    if (
        max_individual_goal_err < 1e-3
        and max_step_speed <= problem.max_speed + 1e-5
        and min_observed_sep >= problem.scenario.min_separation - 1e-5
        and val_result.feasible
    ):
        print(">>> SUCCESS: UAVs TRULY AND SAFELY MOVED TO THE TARGET SOLUTION! <<<")
    else:
        print(">>> FAILURE: Trajectory did not satisfy all constraints. <<<")
    print("=" * 80)


if __name__ == "__main__":
    simulate_and_verify()
