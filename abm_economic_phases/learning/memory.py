"""
Multi-Level Adaptive Memory System

Implements Section 6.1:
- M_micro^i(t+1) = (1 - δ_m) * M_micro^i(t) + δ_m * R_i(t) * exp(-|R_i(t)|/τ)
- M_meso^j(t+1) = (1/|G_j|) * sum_{i∈G_j} M_micro^i(t) + λ_j * Sector_shocks_j
- M_macro(t+1) = tanh(sum_j β_j * M_meso^j(t) + γ * Systemic_events)

Memory affects:
- Tension adjustment (reduces effective tension)
- Absorption capacity for positive shocks
- Learning rate modulation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import numpy as np


@dataclass
class MemoryState:
    """State of the multi-level memory system."""

    # Micro level (individual agents)
    micro_memories: Dict[int, float] = field(default_factory=dict)

    # Meso level (sectors/groups)
    meso_memories: Dict[str, float] = field(default_factory=dict)

    # Macro level (system-wide)
    macro_memory: float = 0.0

    # Aggregates
    avg_micro: float = 0.0
    avg_meso: float = 0.0


class AdaptiveMemorySystem:
    """
    Multi-level adaptive memory system.

    Three levels:
    1. Micro: Individual agent memories (experiences)
    2. Meso: Sector/group aggregated memories
    3. Macro: System-wide collective memory

    Memory serves as:
    - Tension dampener (experienced systems handle stress better)
    - Absorption capacity enhancer for positive shocks
    - Learning rate modulator (experienced agents learn differently)
    """

    def __init__(
        self,
        sectors: Optional[List[str]] = None,
        delta_m: float = 0.05,  # Micro memory decay
        tau: float = 0.1,  # Impact normalization
        lambda_sector: float = 0.2,  # Sector shock contribution
        gamma_systemic: float = 0.4,  # Systemic event weight
        sector_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.sectors = sectors or [
            "primary",
            "manufacturing",
            "services",
            "financial",
            "public",
        ]
        self.delta_m = delta_m
        self.tau = tau
        self.lambda_sector = lambda_sector
        self.gamma_systemic = gamma_systemic
        self.rng = np.random.default_rng(seed)

        # Sector weights for macro aggregation
        self.sector_weights = sector_weights or {
            "primary": 0.15,
            "manufacturing": 0.25,
            "services": 0.35,
            "financial": 0.15,
            "public": 0.10,
        }

        # State
        self.state = MemoryState()

        # Agent-sector mapping
        self.agent_sectors: Dict[int, str] = {}

        # History
        self.memory_history: List[MemoryState] = []

    def register_agent(
        self,
        agent_id: int,
        sector: str = "services",
    ) -> None:
        """Register an agent with the memory system."""
        self.state.micro_memories[agent_id] = 0.0
        self.agent_sectors[agent_id] = sector

        if sector not in self.state.meso_memories:
            self.state.meso_memories[sector] = 0.0

    def update_micro_memory(
        self,
        agent_id: int,
        reward: float,
    ) -> float:
        """
        Update individual agent memory using Equation (14).

        M_micro^i(t+1) = (1 - δ_m) * M_micro^i(t) + δ_m * R_i(t) * exp(-|R_i(t)|/τ)

        The exponential term prevents extreme outcomes from dominating memory.
        """
        if agent_id not in self.state.micro_memories:
            self.register_agent(agent_id)

        current = self.state.micro_memories[agent_id]

        # Memory update with exponential dampening of extreme outcomes
        memory_contribution = reward * np.exp(-abs(reward) / self.tau)

        new_memory = (1 - self.delta_m) * current + self.delta_m * memory_contribution

        # Bound memory to prevent explosion
        new_memory = np.clip(new_memory, -1, 1)

        self.state.micro_memories[agent_id] = new_memory
        return new_memory

    def update_meso_memory(
        self,
        sector_shocks: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Update sector-level memories using Equation (15).

        M_meso^j(t+1) = (1/|G_j|) * sum_{i∈G_j} M_micro^i(t) + λ_j * Sector_shocks_j
        """
        sector_shocks = sector_shocks or {}

        # Compute sector averages
        sector_agents: Dict[str, List[int]] = {s: [] for s in self.sectors}
        for agent_id, sector in self.agent_sectors.items():
            if sector in sector_agents:
                sector_agents[sector].append(agent_id)

        for sector in self.sectors:
            agents = sector_agents.get(sector, [])

            if agents:
                # Average of micro memories in this sector
                avg_micro = np.mean([
                    self.state.micro_memories.get(a, 0) for a in agents
                ])
            else:
                avg_micro = 0.0

            # Add sector-specific shocks
            shock = sector_shocks.get(sector, 0.0)

            new_meso = avg_micro + self.lambda_sector * shock
            new_meso = np.clip(new_meso, -1, 1)

            self.state.meso_memories[sector] = new_meso

        return self.state.meso_memories.copy()

    def update_macro_memory(
        self,
        systemic_event_impact: float = 0.0,
    ) -> float:
        """
        Update system-wide memory using Equation (16).

        M_macro(t+1) = tanh(sum_j β_j * M_meso^j(t) + γ * Systemic_events)

        The tanh ensures macro memory stays bounded in [-1, 1].
        """
        # Weighted sum of sector memories
        weighted_sum = sum(
            self.sector_weights.get(sector, 0.2) * self.state.meso_memories.get(sector, 0)
            for sector in self.sectors
        )

        # Add systemic event contribution
        total = weighted_sum + self.gamma_systemic * systemic_event_impact

        # Apply tanh for bounded output
        self.state.macro_memory = np.tanh(total)

        # Update aggregates
        if self.state.micro_memories:
            self.state.avg_micro = np.mean(list(self.state.micro_memories.values()))
        if self.state.meso_memories:
            self.state.avg_meso = np.mean(list(self.state.meso_memories.values()))

        return self.state.macro_memory

    def update_all(
        self,
        agent_rewards: Dict[int, float],
        sector_shocks: Optional[Dict[str, float]] = None,
        systemic_event_impact: float = 0.0,
    ) -> MemoryState:
        """
        Update all memory levels in sequence.

        Args:
            agent_rewards: Dict mapping agent_id to reward/outcome
            sector_shocks: Optional sector-level shocks
            systemic_event_impact: Impact of any systemic event

        Returns:
            Updated MemoryState
        """
        # Update micro level
        for agent_id, reward in agent_rewards.items():
            self.update_micro_memory(agent_id, reward)

        # Update meso level
        self.update_meso_memory(sector_shocks)

        # Update macro level
        self.update_macro_memory(systemic_event_impact)

        # Store history
        self.memory_history.append(MemoryState(
            micro_memories=self.state.micro_memories.copy(),
            meso_memories=self.state.meso_memories.copy(),
            macro_memory=self.state.macro_memory,
            avg_micro=self.state.avg_micro,
            avg_meso=self.state.avg_meso,
        ))

        return self.state

    def get_tension_adjustment(self) -> float:
        """
        Get the memory-based tension adjustment factor.

        Higher memory = lower effective tension.
        Used in Equation (9): T_adj = sum(w*T) / (1 + λ*M_macro)
        """
        # Positive memory reduces tension, negative increases it
        return max(0, self.state.macro_memory)

    def get_absorption_capacity(self, base_capacity: float = 0.05) -> float:
        """
        Compute absorption capacity for positive shocks.

        Higher memory = better ability to absorb positive shocks.
        Used in Equation (13): κ_abs = κ_0 + γ*M_macro*A
        """
        memory_contribution = max(0, self.state.macro_memory) * 0.3
        return base_capacity + memory_contribution

    def get_learning_rate_modifier(self, agent_id: int) -> float:
        """
        Get a learning rate modifier based on agent experience.

        Experienced agents (high memory) may learn more slowly (exploitation)
        Inexperienced agents (low memory) learn faster (exploration)
        """
        memory = self.state.micro_memories.get(agent_id, 0)

        # Higher memory = slower learning (more exploitation)
        # Range: 0.5 to 1.5
        modifier = 1.0 - 0.5 * memory

        return np.clip(modifier, 0.5, 1.5)

    def get_sector_resilience(self, sector: str) -> float:
        """
        Get sector-level resilience based on accumulated memory.

        Positive memory indicates past success handling shocks.
        """
        memory = self.state.meso_memories.get(sector, 0)
        return 0.5 + 0.5 * max(0, memory)  # Range: 0.5 to 1.0

    def compute_collective_wisdom(self) -> float:
        """
        Compute a measure of collective economic wisdom.

        Combines:
        - Diversity of agent experiences (variance of micro memories)
        - Sector coordination (alignment of meso memories)
        - Overall system learning (macro memory trend)
        """
        if not self.state.micro_memories:
            return 0.5

        # Experience diversity (some variance is good, too much is bad)
        micro_values = list(self.state.micro_memories.values())
        variance = np.var(micro_values)
        diversity_score = 1 - np.exp(-variance * 10)  # Peaks at moderate variance

        # Sector coordination
        meso_values = list(self.state.meso_memories.values())
        if len(meso_values) > 1:
            meso_variance = np.var(meso_values)
            coordination = np.exp(-meso_variance * 5)  # Higher = more coordinated
        else:
            coordination = 0.5

        # Overall learning (positive trend in macro memory)
        if len(self.memory_history) >= 4:
            recent = [h.macro_memory for h in self.memory_history[-4:]]
            trend = (recent[-1] - recent[0]) / 4 if len(recent) >= 2 else 0
            learning_score = 0.5 + 0.5 * np.tanh(trend * 10)
        else:
            learning_score = 0.5

        # Weighted combination
        wisdom = 0.3 * diversity_score + 0.3 * coordination + 0.4 * learning_score

        return np.clip(wisdom, 0, 1)

    def get_memory_vector(self) -> np.ndarray:
        """Get memory as numpy array [M_micro_avg, M_meso_avg, M_macro]."""
        return np.array([
            self.state.avg_micro,
            self.state.avg_meso,
            self.state.macro_memory,
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Export current memory state."""
        return {
            "macro_memory": self.state.macro_memory,
            "avg_micro": self.state.avg_micro,
            "avg_meso": self.state.avg_meso,
            "meso_memories": self.state.meso_memories.copy(),
            "collective_wisdom": self.compute_collective_wisdom(),
        }

    def reset(self) -> None:
        """Reset all memories."""
        self.state = MemoryState()
        self.agent_sectors.clear()
        self.memory_history.clear()
