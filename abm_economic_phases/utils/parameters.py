"""
Model Parameters

Calibrated parameters from the paper (Table 6):
- N_h, N_f, N_b: Number of agents (scalable)
- gamma (network): Degree distribution exponent
- lambda_0 (events): Base rate of extreme events
- kappa_abs_0: Base absorption capacity
- beta (learning): Learning rate
- delta_m (memory): Memory decay rate
- alpha, beta, gamma (MMT): Fiscal policy parameters
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class NetworkParameters:
    """Parameters for the scale-free network topology."""

    gamma: float = 2.3  # Degree distribution exponent (2 < gamma < 3)
    k0: float = 1.0  # Intrinsic attraction parameter
    m: int = 3  # Edges per new node in preferential attachment


@dataclass
class AgentParameters:
    """Parameters for agent behavior."""

    # Number of agents
    n_households: int = 1000
    n_firms: int = 100
    n_banks: int = 10

    # Household parameters
    household_risk_aversion: float = 2.0
    household_discount_factor: float = 0.96
    labor_elasticity: float = 0.5

    # Firm parameters
    firm_markup_mean: float = 0.15
    firm_markup_std: float = 0.05
    capital_depreciation: float = 0.025
    production_alpha: float = 0.33  # Capital share in Cobb-Douglas

    # Bank parameters
    bank_reserve_ratio: float = 0.1
    bank_capital_ratio: float = 0.08
    risk_weight_default: float = 1.0

    # Noise parameters by agent type
    noise_std: Dict[str, float] = field(default_factory=lambda: {
        "household": 0.02,
        "firm": 0.03,
        "bank": 0.01,
    })


@dataclass
class EventParameters:
    """Parameters for extreme events (Black Swans and Unicorns)."""

    lambda_0: float = 0.01  # Base rate of extreme events
    kappa: float = 2.0  # Sensitivity to tension in event rate
    t_crit: float = 0.7  # Critical tension threshold

    # Negative event (Black Swan) parameters
    beta_amplification: float = 1.5  # Impact amplification factor
    alpha_threshold: float = 2.0  # Threshold steepness
    xi_0: float = -0.05  # Critical amplification threshold

    # Positive event (Unicorn) parameters
    kappa_abs_0: float = 0.05  # Base absorption capacity
    gamma_memory: float = 0.3  # Memory contribution to absorption
    sigma_abundance: float = 0.02  # Spread of abundance effect
    phi_threshold: float = 2.5  # "Too much success" threshold
    omega_0: float = 0.01  # Base secondary effects
    omega_1: float = 0.02  # Financial tension contribution
    omega_2: float = 0.03  # Incoherence contribution


@dataclass
class MemoryParameters:
    """Parameters for the multi-level adaptive memory system."""

    delta_m: float = 0.05  # Memory decay rate (micro level)
    tau: float = 0.1  # Impact normalization constant
    lambda_sector: float = 0.2  # Sector shock contribution (meso level)

    # Weights for macro memory aggregation
    sector_weights: Dict[str, float] = field(default_factory=lambda: {
        "primary": 0.15,
        "manufacturing": 0.25,
        "services": 0.35,
        "financial": 0.15,
        "public": 0.10,
    })

    gamma_systemic: float = 0.4  # Systemic events weight in macro memory


@dataclass
class LearningParameters:
    """Parameters for reinforcement learning (SARSA-lambda)."""

    alpha: float = 0.1  # Learning rate
    gamma: float = 0.95  # Discount factor
    lambda_trace: float = 0.8  # Eligibility trace decay
    beta_softmax: float = 5.0  # Softmax temperature
    eta_memory: float = 0.1  # Memory incorporation rate

    # Exploration parameters
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995


@dataclass
class PhaseParameters:
    """Parameters for economic phase transitions."""

    # Hysteresis deltas for each transition
    hysteresis: Dict[str, float] = field(default_factory=lambda: {
        "activation_expansion": 0.005,
        "expansion_maturity": 0.002,
        "maturity_overheating": 0.1,
        "overheating_crisis": 0.05,
        "crisis_recession": 0.03,
        "recession_activation": 0.01,
    })

    # Duration requirements (quarters)
    duration_requirements: Dict[str, int] = field(default_factory=lambda: {
        "activation_expansion": 2,
        "expansion_maturity": 4,
        "crisis_recession": 2,
        "recession_activation": 3,
    })

    # Transition noise
    transition_noise_std: float = 0.1


@dataclass
class MMTParameters:
    """Parameters for MMT (Modern Monetary Theory) compatibility."""

    # Fiscal policy rule: G_t = G_bar - alpha*(Y - Y_pot) + delta*T_adj - gamma*I_crisis
    g_bar: float = 0.2  # Base government spending as GDP fraction
    alpha_gap: float = 0.3  # Response to output gap
    delta_tension: float = 0.1  # Response to tension
    gamma_crisis: float = 0.4  # Crisis intervention multiplier

    # Inflation equation coefficients
    beta_inflation: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.3, 0.1, 0.15, 0.25])
    )  # [constant, output gap, T_E, T_C, expectations]

    # Employer of Last Resort
    employment_target: float = 0.96  # Target employment rate


@dataclass
class ModelParameters:
    """Complete model parameters aggregating all sub-parameters."""

    network: NetworkParameters = field(default_factory=NetworkParameters)
    agents: AgentParameters = field(default_factory=AgentParameters)
    events: EventParameters = field(default_factory=EventParameters)
    memory: MemoryParameters = field(default_factory=MemoryParameters)
    learning: LearningParameters = field(default_factory=LearningParameters)
    phases: PhaseParameters = field(default_factory=PhaseParameters)
    mmt: MMTParameters = field(default_factory=MMTParameters)

    # Simulation parameters
    seed: int = 42
    time_horizon: int = 100  # Quarters

    # Tension weight adaptation
    tension_learning_rate: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert all parameters to a flat dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, '__dataclass_fields__'):
                for sub_field in value.__dataclass_fields__:
                    result[f"{field_name}.{sub_field}"] = getattr(value, sub_field)
            else:
                result[field_name] = value
        return result

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelParameters":
        """Create parameters from a dictionary."""
        params = cls()
        for key, value in config.items():
            if "." in key:
                parent, child = key.split(".", 1)
                if hasattr(params, parent):
                    parent_obj = getattr(params, parent)
                    if hasattr(parent_obj, child):
                        setattr(parent_obj, child, value)
            elif hasattr(params, key):
                setattr(params, key, value)
        return params
