#!/usr/bin/env python3
"""
Basic Simulation Example

Demonstrates how to run the ABM Economic Phases model
with default parameters and visualize results.
"""

import numpy as np
import matplotlib.pyplot as plt
from abm_economic_phases import EconomicSimulation, ModelParameters


def run_basic_simulation():
    """Run a basic simulation with default parameters."""
    print("=" * 60)
    print("ABM Economic Phases Model - Basic Simulation")
    print("=" * 60)

    # Create parameters with smaller agent counts for faster demo
    params = ModelParameters()
    params.agents.n_households = 500
    params.agents.n_firms = 50
    params.agents.n_banks = 5
    params.time_horizon = 100  # 100 quarters = 25 years

    # Initialize simulation
    print("\nInitializing simulation...")
    sim = EconomicSimulation(params=params, seed=42)

    # Run simulation
    print(f"Running {params.time_horizon} time steps...")
    results = sim.run(progress_bar=True)

    # Print summary
    summary = sim.get_summary()
    print("\n" + "=" * 60)
    print("Simulation Summary")
    print("=" * 60)
    print(f"Total steps: {summary['steps']}")
    print(f"Average GDP growth: {summary['avg_gdp_growth']:.2%}")
    print(f"GDP growth volatility: {summary['std_gdp_growth']:.2%}")
    print(f"Average unemployment: {summary['avg_unemployment']:.2%}")
    print(f"Average inflation: {summary['avg_inflation']:.2%}")
    print(f"Number of crises: {summary['n_crises']}")
    print(f"Number of extreme events: {summary['n_events']}")
    print("\nPhase durations (quarters):")
    for phase, duration in summary['phase_durations'].items():
        print(f"  {phase}: {duration}")

    return results, summary


def plot_results(results):
    """Create visualization of simulation results."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('ABM Economic Phases - Simulation Results', fontsize=14)

    quarters = range(len(results.gdp))

    # GDP Growth
    ax = axes[0, 0]
    ax.plot(quarters, [g * 100 for g in results.gdp_growth], 'b-', linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(quarters, 0, [g * 100 for g in results.gdp_growth],
                    where=[g > 0 for g in results.gdp_growth], alpha=0.3, color='green')
    ax.fill_between(quarters, 0, [g * 100 for g in results.gdp_growth],
                    where=[g < 0 for g in results.gdp_growth], alpha=0.3, color='red')
    ax.set_ylabel('GDP Growth (%)')
    ax.set_title('GDP Growth Rate')
    ax.grid(True, alpha=0.3)

    # Unemployment
    ax = axes[0, 1]
    ax.plot(quarters, [u * 100 for u in results.unemployment], 'r-', linewidth=0.8)
    ax.set_ylabel('Unemployment (%)')
    ax.set_title('Unemployment Rate')
    ax.grid(True, alpha=0.3)

    # Inflation
    ax = axes[1, 0]
    ax.plot(quarters, [i * 100 for i in results.inflation], 'purple', linewidth=0.8)
    ax.axhline(y=2, color='k', linestyle='--', alpha=0.3, label='2% target')
    ax.set_ylabel('Inflation (%)')
    ax.set_title('Inflation Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tensions
    ax = axes[1, 1]
    t_adjusted = [t.get('t_adjusted', 0) for t in results.tensions]
    t_financial = [t.get('t_financial', 0) for t in results.tensions]
    ax.plot(quarters, t_adjusted, 'orange', linewidth=0.8, label='Adjusted')
    ax.plot(quarters, t_financial, 'red', linewidth=0.8, alpha=0.5, label='Financial')
    ax.set_ylabel('Tension Index')
    ax.set_title('Systemic Tensions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phases
    ax = axes[2, 0]
    phase_map = {
        'activation': 0, 'expansion': 1, 'maturity': 2,
        'overheating': 3, 'crisis': 4, 'recession': 5
    }
    phase_values = [phase_map.get(p, 0) for p in results.phases]
    ax.plot(quarters, phase_values, 'k-', linewidth=0.8)
    ax.set_yticks(list(phase_map.values()))
    ax.set_yticklabels(list(phase_map.keys()))
    ax.set_ylabel('Phase')
    ax.set_title('Economic Phases')
    ax.grid(True, alpha=0.3)

    # Memory
    ax = axes[2, 1]
    macro_memory = [m.get('macro_memory', 0) for m in results.memory]
    ax.plot(quarters, macro_memory, 'green', linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('Macro Memory')
    ax.set_title('System Memory')
    ax.set_xlabel('Quarter')
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Quarter')

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'simulation_results.png'")
    plt.show()


def main():
    """Main entry point."""
    results, summary = run_basic_simulation()

    try:
        plot_results(results)
    except Exception as e:
        print(f"\nCould not create plots: {e}")
        print("Results are still available in the 'results' object.")


if __name__ == "__main__":
    main()
