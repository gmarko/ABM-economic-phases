"""
Economic Phases and Transitions

Implements the phase space from Section 3.3 and transition mechanisms from Section 7:
- F_t ∈ {Activation, Expansion, Maturity, Overheating, Crisis, Recession}
- Transition conditions from Table 5
- Master transition equation (15): P = 1/(1 + exp(-[sum_i(beta_i * C_i(t)) - theta + epsilon_t]))
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class EconomicPhase(Enum):
    """Economic phases of the business cycle."""

    ACTIVATION = "activation"
    EXPANSION = "expansion"
    MATURITY = "maturity"
    OVERHEATING = "overheating"
    CRISIS = "crisis"
    RECESSION = "recession"


@dataclass
class PhaseConditions:
    """Conditions for phase transitions from Table 5."""

    # GDP growth thresholds
    g_t: float = 0.0

    # GDP acceleration
    a_t: float = 0.0

    # Sectoral coherence
    theta_t: float = 0.7

    # Tension indices
    t_adjusted: float = 0.0
    t_f: float = 0.0  # Financial tension
    t_e: float = 0.0  # Energy tension

    # Memory
    m_macro: float = 0.0

    # Other indicators
    capacity_utilization: float = 0.85
    duration_in_phase: int = 0  # Quarters in current phase


@dataclass
class TransitionRule:
    """Rule for a specific phase transition."""

    from_phase: EconomicPhase
    to_phase: EconomicPhase

    # Primary condition (must be satisfied)
    primary_condition: str  # Condition expression
    primary_threshold: float

    # Secondary condition (must also be satisfied)
    secondary_condition: Optional[str] = None
    secondary_threshold: Optional[float] = None

    # Duration requirement (quarters)
    duration_requirement: int = 1

    # Hysteresis (prevents rapid switching back)
    hysteresis: float = 0.0

    # Weight in probability calculation
    beta: float = 1.0


class PhaseTransitionEngine:
    """
    Engine for computing economic phase transitions.

    Implements the master transition equation:
    P = 1 / (1 + exp(-[sum_i(beta_i * C_i(t)) - theta + epsilon_t]))

    where C_i(t) are the conditions from Table 5.
    """

    def __init__(
        self,
        transition_noise_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.noise_std = transition_noise_std
        self.rng = np.random.default_rng(seed)

        # Initialize transition rules from Table 5
        self.rules = self._initialize_rules()

        # Track phase history for hysteresis
        self.phase_history: List[EconomicPhase] = []
        self.time_in_current_phase: int = 0

    def _initialize_rules(self) -> Dict[Tuple[EconomicPhase, EconomicPhase], TransitionRule]:
        """Initialize transition rules from Table 5."""
        rules = {}

        # Activation → Expansion
        rules[(EconomicPhase.ACTIVATION, EconomicPhase.EXPANSION)] = TransitionRule(
            from_phase=EconomicPhase.ACTIVATION,
            to_phase=EconomicPhase.EXPANSION,
            primary_condition="g_t > threshold",
            primary_threshold=0.02,
            secondary_condition="theta_t > threshold",
            secondary_threshold=0.5,
            duration_requirement=2,
            hysteresis=0.005,
            beta=1.0,
        )

        # Expansion → Maturity
        rules[(EconomicPhase.EXPANSION, EconomicPhase.MATURITY)] = TransitionRule(
            from_phase=EconomicPhase.EXPANSION,
            to_phase=EconomicPhase.MATURITY,
            primary_condition="|a_t| < threshold",
            primary_threshold=0.001,
            secondary_condition="t_adjusted < threshold",
            secondary_threshold=0.3,
            duration_requirement=4,
            hysteresis=0.002,
            beta=1.0,
        )

        # Maturity → Overheating
        rules[(EconomicPhase.MATURITY, EconomicPhase.OVERHEATING)] = TransitionRule(
            from_phase=EconomicPhase.MATURITY,
            to_phase=EconomicPhase.OVERHEATING,
            primary_condition="t_f > threshold or t_e > threshold",
            primary_threshold=0.6,  # For t_f; t_e uses 0.7
            secondary_condition="theta_t < threshold",
            secondary_threshold=0.6,
            duration_requirement=1,
            hysteresis=0.1,
            beta=1.5,
        )

        # Overheating → Crisis
        rules[(EconomicPhase.OVERHEATING, EconomicPhase.CRISIS)] = TransitionRule(
            from_phase=EconomicPhase.OVERHEATING,
            to_phase=EconomicPhase.CRISIS,
            primary_condition="g_t < 0 and a_t < threshold",
            primary_threshold=-0.01,
            secondary_condition="t_adjusted > threshold",
            secondary_threshold=0.8,
            duration_requirement=1,
            hysteresis=0.05,
            beta=2.0,
        )

        # Crisis → Recession
        rules[(EconomicPhase.CRISIS, EconomicPhase.RECESSION)] = TransitionRule(
            from_phase=EconomicPhase.CRISIS,
            to_phase=EconomicPhase.RECESSION,
            primary_condition="a_t > threshold",
            primary_threshold=0.0,
            secondary_condition="m_macro > threshold",
            secondary_threshold=0.4,
            duration_requirement=2,
            hysteresis=0.03,
            beta=1.0,
        )

        # Recession → Activation
        rules[(EconomicPhase.RECESSION, EconomicPhase.ACTIVATION)] = TransitionRule(
            from_phase=EconomicPhase.RECESSION,
            to_phase=EconomicPhase.ACTIVATION,
            primary_condition="g_t > threshold",
            primary_threshold=0.0,
            secondary_condition="capacity_slack > threshold",
            secondary_threshold=0.15,
            duration_requirement=3,
            hysteresis=0.01,
            beta=1.0,
        )

        # Direct crisis transitions (from any stable phase)
        for phase in [EconomicPhase.EXPANSION, EconomicPhase.MATURITY]:
            rules[(phase, EconomicPhase.CRISIS)] = TransitionRule(
                from_phase=phase,
                to_phase=EconomicPhase.CRISIS,
                primary_condition="shock",
                primary_threshold=0.0,
                duration_requirement=1,
                hysteresis=0.05,
                beta=3.0,  # High weight for sudden crises
            )

        return rules

    def evaluate_condition(
        self,
        condition: str,
        threshold: float,
        conditions: PhaseConditions,
    ) -> Tuple[bool, float]:
        """
        Evaluate a transition condition.

        Returns (is_satisfied, condition_value).
        """
        if condition == "g_t > threshold":
            return conditions.g_t > threshold, conditions.g_t

        elif condition == "|a_t| < threshold":
            abs_a = abs(conditions.a_t)
            return abs_a < threshold, -abs_a  # Negative so smaller is better

        elif condition == "t_f > threshold or t_e > threshold":
            val = max(conditions.t_f, conditions.t_e)
            return val > threshold or conditions.t_e > 0.7, val

        elif condition == "theta_t > threshold":
            return conditions.theta_t > threshold, conditions.theta_t

        elif condition == "theta_t < threshold":
            return conditions.theta_t < threshold, -conditions.theta_t

        elif condition == "t_adjusted < threshold":
            return conditions.t_adjusted < threshold, -conditions.t_adjusted

        elif condition == "t_adjusted > threshold":
            return conditions.t_adjusted > threshold, conditions.t_adjusted

        elif condition == "g_t < 0 and a_t < threshold":
            satisfied = conditions.g_t < 0 and conditions.a_t < threshold
            return satisfied, -(conditions.g_t + conditions.a_t)

        elif condition == "a_t > threshold":
            return conditions.a_t > threshold, conditions.a_t

        elif condition == "m_macro > threshold":
            return conditions.m_macro > threshold, conditions.m_macro

        elif condition == "capacity_slack > threshold":
            slack = 1 - conditions.capacity_utilization
            return slack > threshold, slack

        elif condition == "shock":
            # External shock trigger (handled separately)
            return False, 0.0

        return False, 0.0

    def compute_transition_probability(
        self,
        current_phase: EconomicPhase,
        conditions: PhaseConditions,
    ) -> Dict[EconomicPhase, float]:
        """
        Compute transition probabilities to all possible next phases.

        Uses master equation (15):
        P = 1 / (1 + exp(-[sum_i(beta_i * C_i(t)) - theta + epsilon_t]))
        """
        probabilities = {}

        # Get all valid transitions from current phase
        valid_transitions = [
            (to_phase, rule)
            for (from_phase, to_phase), rule in self.rules.items()
            if from_phase == current_phase
        ]

        for to_phase, rule in valid_transitions:
            # Check duration requirement
            if conditions.duration_in_phase < rule.duration_requirement:
                probabilities[to_phase] = 0.0
                continue

            # Evaluate primary condition
            primary_satisfied, primary_value = self.evaluate_condition(
                rule.primary_condition, rule.primary_threshold, conditions
            )

            if not primary_satisfied:
                probabilities[to_phase] = 0.0
                continue

            # Evaluate secondary condition if exists
            if rule.secondary_condition and rule.secondary_threshold is not None:
                secondary_satisfied, _ = self.evaluate_condition(
                    rule.secondary_condition, rule.secondary_threshold, conditions
                )
                if not secondary_satisfied:
                    probabilities[to_phase] = 0.0
                    continue

            # Compute probability using logistic function
            # Higher primary_value = higher probability
            z = rule.beta * primary_value - rule.hysteresis
            noise = self.rng.normal(0, self.noise_std)
            z += noise

            prob = 1 / (1 + np.exp(-z))
            probabilities[to_phase] = np.clip(prob, 0, 1)

        # Probability of staying in current phase
        total_transition_prob = sum(probabilities.values())
        probabilities[current_phase] = max(0, 1 - total_transition_prob)

        # Normalize if needed
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}

        return probabilities

    def transition(
        self,
        current_phase: EconomicPhase,
        conditions: PhaseConditions,
        force_crisis: bool = False,
    ) -> Tuple[EconomicPhase, Dict[EconomicPhase, float]]:
        """
        Determine the next phase based on conditions.

        Args:
            current_phase: Current economic phase
            conditions: Current phase conditions
            force_crisis: If True, force transition to crisis (for black swan events)

        Returns:
            Tuple of (new_phase, transition_probabilities)
        """
        if force_crisis:
            self.time_in_current_phase = 0
            self.phase_history.append(EconomicPhase.CRISIS)
            return EconomicPhase.CRISIS, {EconomicPhase.CRISIS: 1.0}

        # Compute probabilities
        probabilities = self.compute_transition_probability(current_phase, conditions)

        # Sample next phase
        phases = list(probabilities.keys())
        probs = list(probabilities.values())

        if sum(probs) > 0:
            next_phase = self.rng.choice(phases, p=probs)
        else:
            next_phase = current_phase

        # Update tracking
        if next_phase != current_phase:
            self.time_in_current_phase = 0
        else:
            self.time_in_current_phase += 1

        self.phase_history.append(next_phase)

        return next_phase, probabilities

    def get_phase_characteristics(
        self,
        phase: EconomicPhase,
    ) -> Dict[str, Any]:
        """
        Get characteristic ranges for a phase from Table 2.

        Returns expected ranges for g_t, a_t, theta_t and interpretation.
        """
        characteristics = {
            EconomicPhase.ACTIVATION: {
                "g_t_range": (0.0, 0.02),
                "a_t_sign": "positive",
                "theta_t_range": (0.3, 0.6),
                "interpretation": "Incipient recovery, leading sectors emerge",
            },
            EconomicPhase.EXPANSION: {
                "g_t_range": (0.02, 0.05),
                "a_t_sign": "positive",
                "theta_t_range": (0.6, 0.9),
                "interpretation": "Sustained and coordinated growth",
            },
            EconomicPhase.MATURITY: {
                "g_t_range": (0.02, 0.04),
                "a_t_sign": "zero",
                "theta_t_range": (0.7, 0.95),
                "interpretation": "Stability, diminishing marginal returns",
            },
            EconomicPhase.OVERHEATING: {
                "g_t_range": (0.05, float("inf")),
                "a_t_sign": "negative",
                "theta_t_range": (0.4, 0.7),
                "interpretation": "Uncoordinated growth, sectoral bubbles",
            },
            EconomicPhase.CRISIS: {
                "g_t_range": (float("-inf"), 0.0),
                "a_t_sign": "negative",
                "theta_t_range": (0.1, 0.4),
                "interpretation": "Generalized contraction, loss of confidence",
            },
            EconomicPhase.RECESSION: {
                "g_t_range": (-0.03, 0.0),
                "a_t_sign": "positive",
                "theta_t_range": (0.2, 0.5),
                "interpretation": "End of contraction, structural adjustments",
            },
        }

        return characteristics.get(phase, {})

    def reset(self) -> None:
        """Reset the transition engine."""
        self.phase_history.clear()
        self.time_in_current_phase = 0
