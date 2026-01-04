"""
Base Agent Class

Implements the fundamental agent dynamics from the paper (Equation 3):
a_i^{t+1} = Phi_tau(a_i^t, S_t, I_i^t, epsilon_i^t, M_i^t)

Each agent has:
- Action vector (consumption, investment, labor supply, etc.)
- Access to macroeconomic state S_t
- Information set I_i^t (local and filtered global)
- Idiosyncratic noise epsilon_i^t ~ N(0, sigma_tau^2)
- Individual accumulated memory M_i^t
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import numpy as np


class AgentType(Enum):
    """Types of agents in the model."""

    HOUSEHOLD = "household"
    FIRM = "firm"
    BANK = "bank"
    GOVERNMENT = "government"


@dataclass
class AgentState:
    """State of an individual agent at time t."""

    # Economic variables
    wealth: float = 0.0
    income: float = 0.0
    debt: float = 0.0

    # Activity indicators
    is_active: bool = True
    is_employed: bool = True  # For households
    is_bankrupt: bool = False

    # Memory and learning
    memory: float = 0.0  # M_i^t
    q_values: Dict[str, float] = field(default_factory=dict)
    eligibility_traces: Dict[str, float] = field(default_factory=dict)

    # Network position
    degree: int = 0
    neighbors: Set[int] = field(default_factory=set)


@dataclass
class MacroState:
    """Aggregate macroeconomic state S_t = (F_t, T_t, A_t, M_t)."""

    phase: str = "activation"  # F_t
    tensions: np.ndarray = field(
        default_factory=lambda: np.zeros(5)
    )  # T_t = [T_E, T_C, T_D, T_F, T_X]
    coupling: float = 0.5  # A_t in [0, 1]
    memory: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # M_t = [M_micro, M_meso, M_macro]

    # Aggregate indicators
    gdp: float = 1.0
    gdp_growth: float = 0.02
    gdp_acceleration: float = 0.0
    sectoral_coherence: float = 0.7
    unemployment: float = 0.05
    inflation: float = 0.02
    interest_rate: float = 0.03
    potential_output: float = 1.0


class Agent(ABC):
    """
    Abstract base class for all agents.

    Implements core dynamics from Equation (3):
    a_i^{t+1} = Phi_tau(a_i^t, S_t, I_i^t, epsilon_i^t, M_i^t)
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: AgentType,
        noise_std: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Initialize state
        self.state = AgentState()
        self._initialize_state()

        # Action history for learning
        self.action_history: List[Dict[str, float]] = []
        self.reward_history: List[float] = []

    @abstractmethod
    def _initialize_state(self) -> None:
        """Initialize agent-specific state variables."""
        pass

    @abstractmethod
    def decide(
        self,
        macro_state: MacroState,
        neighbor_info: Dict[int, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Make decisions based on current state and information.

        This implements the function Phi_tau.

        Args:
            macro_state: Current macroeconomic state S_t
            neighbor_info: Information from network neighbors I_i^t

        Returns:
            Dictionary of action variables
        """
        pass

    @abstractmethod
    def update(
        self,
        actions: Dict[str, float],
        macro_state: MacroState,
        market_outcomes: Dict[str, float],
    ) -> None:
        """
        Update agent state after market clearing.

        Args:
            actions: Actions taken this period
            macro_state: Current macroeconomic state
            market_outcomes: Results from market clearing
        """
        pass

    def get_noise(self) -> float:
        """Generate idiosyncratic noise epsilon_i^t ~ N(0, sigma_tau^2)."""
        return self.rng.normal(0, self.noise_std)

    def update_memory(
        self,
        reward: float,
        tau: float = 0.1,
        delta_m: float = 0.05,
    ) -> None:
        """
        Update individual memory according to Equation (14):
        M_micro^i(t+1) = (1 - delta_m) * M_micro^i(t) + delta_m * R_i(t) * exp(-|R_i(t)|/tau)
        """
        memory_update = reward * np.exp(-abs(reward) / tau)
        self.state.memory = (1 - delta_m) * self.state.memory + delta_m * memory_update
        self.reward_history.append(reward)

    def get_info(self) -> Dict[str, Any]:
        """Get information to share with neighbors."""
        return {
            "type": self.agent_type.value,
            "wealth": self.state.wealth,
            "is_active": self.state.is_active,
            "memory": self.state.memory,
        }

    def aggregate_neighbor_info(
        self,
        neighbor_info: Dict[int, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Aggregate information from neighbors.

        Returns averaged signals from the local network neighborhood.
        """
        if not neighbor_info:
            return {"avg_wealth": 0.0, "avg_memory": 0.0, "active_ratio": 0.0}

        active_neighbors = [
            info for info in neighbor_info.values() if info.get("is_active", True)
        ]

        if not active_neighbors:
            return {"avg_wealth": 0.0, "avg_memory": 0.0, "active_ratio": 0.0}

        return {
            "avg_wealth": np.mean([n["wealth"] for n in active_neighbors]),
            "avg_memory": np.mean([n["memory"] for n in active_neighbors]),
            "active_ratio": len(active_neighbors) / len(neighbor_info),
        }

    def compute_reward(
        self,
        actions: Dict[str, float],
        outcomes: Dict[str, float],
    ) -> float:
        """
        Compute reward signal for reinforcement learning.

        Default implementation: change in wealth.
        Override in subclasses for type-specific rewards.
        """
        return outcomes.get("wealth_change", 0.0)

    def reset(self) -> None:
        """Reset agent to initial state."""
        self._initialize_state()
        self.action_history.clear()
        self.reward_history.clear()

    def __repr__(self) -> str:
        return f"{self.agent_type.value.capitalize()}(id={self.agent_id})"
