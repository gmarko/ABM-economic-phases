#!/usr/bin/env python3
"""
Historical Validation Example

Demonstrates how to calibrate the model with historical events
from Table 7 in the paper (2000-2024).
"""

import numpy as np
from abm_economic_phases import EconomicSimulation, ModelParameters
from abm_economic_phases.economy.events import EventType


def run_historical_validation():
    """Run simulation with historical event injection for validation."""
    print("=" * 60)
    print("ABM Economic Phases Model - Historical Validation")
    print("=" * 60)

    # Configure for historical validation
    params = ModelParameters()
    params.agents.n_households = 500
    params.agents.n_firms = 50
    params.agents.n_banks = 5
    params.time_horizon = 100  # 2000-2024 = 25 years = 100 quarters

    # Initialize
    sim = EconomicSimulation(params=params, seed=2000)

    # Define historical events and their approximate quarters
    historical_events = [
        # (quarter, event_name, description)
        (8, "dotcom_crisis", "Dot-com bubble burst"),
        (32, "subprime_crisis", "Subprime mortgage crisis"),
        (40, "eurozone_crisis", "European sovereign debt crisis"),
        (80, "covid_pandemic", "COVID-19 pandemic"),
        (88, "ukraine_war", "Ukraine conflict energy shock"),
    ]

    print("\nRunning simulation with historical event calibration...")
    print("-" * 60)

    crisis_periods = []
    event_impacts = []

    for step in range(params.time_horizon):
        # Check if we should inject a historical event
        for event_quarter, event_name, description in historical_events:
            if step == event_quarter:
                print(f"\nQuarter {step}: Injecting {description}")
                event = sim.event_generator.inject_historical_event(
                    event_name, step
                )
                if event:
                    print(f"  Type: {event.event_type.value}")
                    print(f"  Magnitude: {event.magnitude:.2%}")

        # Run simulation step
        result = sim.step()

        # Track crises
        if result['phase'] == 'crisis':
            crisis_periods.append(step)

        # Track events
        if result['event']:
            event_impacts.append({
                'quarter': step,
                'type': result['event'],
                'gdp_growth': result['gdp_growth'],
            })

    # Print results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    summary = sim.get_summary()
    print(f"\nSimulation covered {summary['steps']} quarters (2000-2024)")
    print(f"Total crisis quarters: {summary['n_crises']}")
    print(f"Total extreme events: {summary['n_events']}")

    # Compare with historical crisis periods
    print("\n--- Crisis Period Analysis ---")
    historical_crisis_quarters = [
        (8, 16),   # Dot-com recession: 2001-2002
        (32, 40),  # Great Recession: 2008-2010
        (80, 84),  # COVID recession: 2020
    ]

    for start, end in historical_crisis_quarters:
        model_crisis_in_period = sum(
            1 for q in crisis_periods if start <= q <= end
        )
        print(f"  Q{start}-Q{end}: {model_crisis_in_period} crisis quarters in model")

    # Performance metrics from Section 9
    print("\n--- Model Performance Metrics ---")

    # GDP correlation would require real data, show simulated metrics
    gdp_growth = sim.results.gdp_growth
    print(f"Average GDP growth: {np.mean(gdp_growth):.3f}")
    print(f"GDP growth std: {np.std(gdp_growth):.3f}")

    # Phase accuracy (simplified)
    phases = sim.results.phases
    phase_counts = {p: phases.count(p) for p in set(phases)}
    print(f"\nPhase distribution:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count} quarters ({count/len(phases)*100:.1f}%)")

    return sim.results, summary


def analyze_event_propagation(results):
    """Analyze how events propagated through the system."""
    print("\n" + "=" * 60)
    print("Event Propagation Analysis")
    print("=" * 60)

    events = [e for e in results.events if e is not None]
    print(f"\nTotal events: {len(events)}")

    if events:
        black_swans = [e for e in events if e['type'] == 'black_swan']
        unicorns = [e for e in events if e['type'] == 'unicorn']

        print(f"Black Swans: {len(black_swans)}")
        print(f"Unicorns: {len(unicorns)}")

        if black_swans:
            avg_bs_impact = np.mean([e['effective_impact'] for e in black_swans])
            print(f"Average Black Swan impact: {avg_bs_impact:.2%}")

        if unicorns:
            avg_u_impact = np.mean([e['effective_impact'] for e in unicorns])
            print(f"Average Unicorn impact: {avg_u_impact:.2%}")


def main():
    """Main entry point."""
    results, summary = run_historical_validation()
    analyze_event_propagation(results)

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
