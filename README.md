# ABM Economic Phases Model

An Agent-Based Model (ABM) for simulating coupled economic phases with external unpredictable variables.

## Overview

This model implements a complex adaptive systems approach to macroeconomic dynamics, featuring:

- **Heterogeneous Agents**: Households, Firms, Banks, and Government with distinct behavioral rules
- **Scale-Free Network Topology**: Preferential attachment for realistic economic interactions
- **Emergent Economic Phases**: Activation, Expansion, Maturity, Overheating, Crisis, Recession
- **GDP Directional Vector**: Three-dimensional representation (growth, acceleration, coherence)
- **Structural Tensions**: Energy, Trade, Currency, Financial, and Event tensions
- **Extreme Events**: Black Swans (negative) and Unicorns (positive) with abundance paradox
- **Adaptive Memory**: Multi-level learning at micro, meso, and macro scales
- **MMT Compatibility**: Modern Monetary Theory fiscal/monetary policy framework

## Installation

```bash
# Clone the repository
git clone https://github.com/mduran/ABM-economic-phases.git
cd ABM-economic-phases

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from abm_economic_phases import EconomicSimulation, ModelParameters

# Create parameters
params = ModelParameters()
params.agents.n_households = 500
params.agents.n_firms = 50
params.agents.n_banks = 5
params.time_horizon = 100  # quarters

# Run simulation
sim = EconomicSimulation(params=params, seed=42)
results = sim.run()

# Get summary
print(sim.get_summary())
```

## Model Architecture

### Agents

| Agent | Function | Behavioral Rule |
|-------|----------|-----------------|
| **Households** | Consumption, Saving, Labor | Utility maximization with adaptive expectations |
| **Firms** | Production, Investment, Employment | Profit maximization with Tobin's Q investment |
| **Banks** | Credit, Intermediation | Risk-based lending with capital constraints |
| **Government** | Fiscal/Monetary Policy | Countercyclical stabilization (MMT compatible) |

### Economic Phases

The model generates endogenous business cycles through phase transitions:

```
Activation → Expansion → Maturity → Overheating → Crisis → Recession → Activation
```

Each phase is characterized by specific ranges of:
- Growth rate (g_t)
- Acceleration (a_t)
- Sectoral coherence (θ_t)

### Key Equations

**GDP Vector:**
```
v_PIB(t) = (g_t, a_t, θ_t)
```

**Adjusted Tension:**
```
T_adjusted(t) = Σ_i w_i(t)·T_i(t) / (1 + λ·M_macro(t))
```

**Phase Transition Probability:**
```
P = 1 / (1 + exp(-[Σ_i β_i·C_i(t) - θ + ε_t]))
```

## Examples

### Basic Simulation
```bash
python examples/basic_simulation.py
```

### Historical Validation (2000-2024)
```bash
python examples/historical_validation.py
```

### Scenario Analysis
```bash
python examples/scenario_analysis.py
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=abm_economic_phases
```

## Project Structure

```
ABM-economic-phases/
├── abm_economic_phases/
│   ├── agents/           # Agent implementations
│   ├── network/          # Scale-free network topology
│   ├── economy/          # Phases, GDP vector, tensions, events
│   ├── learning/         # Memory and RL modules
│   ├── mmt/              # MMT stabilizers
│   ├── simulation/       # Main simulation engine
│   └── utils/            # Parameters and utilities
├── examples/             # Example scripts
├── tests/                # Unit tests
├── requirements.txt
└── setup.py
```

## Configuration

Key parameters can be adjusted via `ModelParameters`:

```python
params = ModelParameters()

# Agent counts
params.agents.n_households = 1000
params.agents.n_firms = 100
params.agents.n_banks = 10

# Network topology
params.network.gamma = 2.3  # Power-law exponent
params.network.m = 3        # Edges per new node

# Event dynamics
params.events.lambda_0 = 0.01  # Base event rate

# Learning
params.learning.alpha = 0.1    # Learning rate
params.memory.delta_m = 0.05   # Memory decay

# MMT policy
params.mmt.g_bar = 0.2         # Base spending
params.mmt.alpha_gap = 0.3     # Output gap response
```

## Validation

The model has been calibrated with historical events (2000-2024):
- Dot-com crisis (2000-2002)
- Subprime crisis (2008-2009)
- Eurozone crisis (2010-2012)
- COVID-19 pandemic (2020)
- Post-COVID inflation (2022-2023)

Performance metrics:
- GDP correlation: ρ = 0.87
- Phase accuracy: 79%
- Directional predictability (3m): Precision = 0.71, Recall = 0.68

## References

Based on the paper:
> Durán Cabobianco, M. (2026). "Modelo de Fases Económicas Acopladas con Variables Externas Impredecibles: Un enfoque basado en sistemas complejos, topologías libres de escala y aprendizaje adaptativo"

## License

MIT License

## Author

Marco Durán Cabobianco
marco@anachroni.co
