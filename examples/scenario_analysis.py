#!/usr/bin/env python3
"""
Scenario Analysis Example

Demonstrates how to run multiple scenarios with different
initial conditions as described in Table 9 (2026-2030 scenarios).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from abm_economic_phases import EconomicSimulation, ModelParameters


@dataclass
class Scenario:
    """A scenario configuration."""
    name: str
    description: str
    initial_tensions: Dict[str, float]
    initial_growth: float
    initial_coupling: float
    policy_adjustments: Dict[str, float]


def create_scenarios() -> List[Scenario]:
    """Create scenarios from Table 9."""
    return [
        Scenario(
            name="Energy Crisis",
            description="High energy tension with moderate growth",
            initial_tensions={"t_energy": 0.7, "t_financial": 0.3},
            initial_growth=0.03,
            initial_coupling=0.7,
            policy_adjustments={"g_bar": 0.22},
        ),
        Scenario(
            name="Financial Stability",
            description="High financial tension with high memory",
            initial_tensions={"t_financial": 0.6, "t_trade": 0.3},
            initial_growth=0.02,
            initial_coupling=0.6,
            policy_adjustments={"alpha_gap": 0.4},
        ),
        Scenario(
            name="Positive Shock",
            description="Unicorn event with high coupling",
            initial_tensions={"t_energy": 0.2, "t_financial": 0.2},
            initial_growth=0.04,
            initial_coupling=0.9,
            policy_adjustments={},
        ),
        Scenario(
            name="Systemic Fragility",
            description="Multiple high tensions",
            initial_tensions={
                "t_energy": 0.5,
                "t_financial": 0.6,
                "t_trade": 0.5,
            },
            initial_growth=0.01,
            initial_coupling=0.5,
            policy_adjustments={"gamma_crisis": 0.5},
        ),
    ]


def run_scenario(scenario: Scenario, seed: int = 42) -> Dict[str, Any]:
    """Run a single scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario.name}")
    print(f"{'='*60}")
    print(f"Description: {scenario.description}")

    # Configure parameters
    params = ModelParameters()
    params.agents.n_households = 300
    params.agents.n_firms = 30
    params.agents.n_banks = 3
    params.time_horizon = 20  # 5 years = 20 quarters

    # Apply policy adjustments
    for key, value in scenario.policy_adjustments.items():
        if hasattr(params.mmt, key):
            setattr(params.mmt, key, value)

    # Initialize simulation
    sim = EconomicSimulation(params=params, seed=seed)

    # Set initial conditions
    for tension_name, value in scenario.initial_tensions.items():
        if hasattr(sim.tension_system.metrics, tension_name):
            setattr(sim.tension_system.metrics, tension_name, value)

    sim.macro_state.gdp_growth = scenario.initial_growth
    sim.macro_state.coupling = scenario.initial_coupling

    # Run simulation
    print(f"\nRunning {params.time_horizon} quarters...")
    results = sim.run(progress_bar=False)

    # Analyze results
    summary = sim.get_summary()

    # Compute trajectory probabilities (simplified)
    final_phase = results.phases[-1]
    n_crises = summary['n_crises']
    avg_growth = summary['avg_gdp_growth']

    trajectory = determine_trajectory(final_phase, n_crises, avg_growth)

    print(f"\n--- Results ---")
    print(f"Final phase: {final_phase}")
    print(f"Average growth: {avg_growth:.2%}")
    print(f"Crisis quarters: {n_crises}")
    print(f"Trajectory: {trajectory}")

    return {
        "scenario": scenario.name,
        "summary": summary,
        "trajectory": trajectory,
        "results": results,
    }


def determine_trajectory(
    final_phase: str,
    n_crises: int,
    avg_growth: float,
) -> str:
    """Determine the trajectory category based on outcomes."""
    if n_crises > 5:
        return "Multiple crises"
    elif final_phase == "crisis":
        return "Crisis transition"
    elif avg_growth < 0.01:
        return "Stagnation"
    elif avg_growth > 0.03:
        return "Sustained growth"
    else:
        return "Moderate stability"


def run_monte_carlo(
    scenario: Scenario,
    n_runs: int = 10,
) -> Dict[str, Any]:
    """Run Monte Carlo analysis for a scenario."""
    print(f"\n{'='*60}")
    print(f"Monte Carlo Analysis: {scenario.name}")
    print(f"{'='*60}")
    print(f"Running {n_runs} simulations...")

    trajectory_counts = {}
    growth_rates = []
    crisis_counts = []

    for i in range(n_runs):
        result = run_scenario(scenario, seed=42 + i)
        trajectory = result['trajectory']
        trajectory_counts[trajectory] = trajectory_counts.get(trajectory, 0) + 1
        growth_rates.append(result['summary']['avg_gdp_growth'])
        crisis_counts.append(result['summary']['n_crises'])

    # Compute probabilities
    print(f"\n--- Trajectory Probabilities ---")
    for trajectory, count in sorted(trajectory_counts.items()):
        prob = count / n_runs
        print(f"  {trajectory}: {prob:.0%}")

    print(f"\n--- Statistics ---")
    print(f"  Growth: {np.mean(growth_rates):.2%} ± {np.std(growth_rates):.2%}")
    print(f"  Crises: {np.mean(crisis_counts):.1f} ± {np.std(crisis_counts):.1f}")

    return {
        "scenario": scenario.name,
        "trajectory_probabilities": {
            k: v / n_runs for k, v in trajectory_counts.items()
        },
        "avg_growth": np.mean(growth_rates),
        "std_growth": np.std(growth_rates),
        "avg_crises": np.mean(crisis_counts),
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("ABM Economic Phases Model - Scenario Analysis")
    print("=" * 60)

    scenarios = create_scenarios()

    # Run each scenario once
    print("\n\n### SINGLE RUN ANALYSIS ###")
    single_results = []
    for scenario in scenarios:
        result = run_scenario(scenario)
        single_results.append(result)

    # Run Monte Carlo for selected scenario
    print("\n\n### MONTE CARLO ANALYSIS ###")
    mc_result = run_monte_carlo(scenarios[0], n_runs=5)

    # Summary comparison
    print("\n\n" + "=" * 60)
    print("Scenario Comparison Summary")
    print("=" * 60)
    print(f"{'Scenario':<20} {'Trajectory':<20} {'Avg Growth':>12}")
    print("-" * 60)
    for result in single_results:
        print(f"{result['scenario']:<20} {result['trajectory']:<20} {result['summary']['avg_gdp_growth']:>12.2%}")


if __name__ == "__main__":
    main()
