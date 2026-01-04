"""
Extreme Events: Black Swans and Unicorns

Implements Section 5.2:
- Non-homogeneous Poisson process for event arrival
- Black Swan (negative) impact function with threshold amplification
- Unicorn (positive) impact with abundance paradox

Key equations:
- Event rate: λ(t) = λ_0 * [1 + κ * tanh(T_adjusted(t) / T_crit)]
- Negative impact: Impact_neg = ξ * [1 + β * T_adj / (1 + exp(-α(ξ - ξ_0)))]
- Positive impact: Impact_pos = ξ * exp(-(ξ - κ_abs)²/(2σ²)) - Ω * I{ξ > φ*κ_abs}
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class EventType(Enum):
    """Types of extreme events."""

    BLACK_SWAN = "black_swan"  # Negative shock (ξ < 0)
    UNICORN = "unicorn"  # Positive shock (ξ > 0)
    GREY_RHINO = "grey_rhino"  # Predictable but ignored risk
    NONE = "none"


@dataclass
class ExtremeEvent:
    """Representation of an extreme event."""

    event_type: EventType
    magnitude: float  # ξ, raw shock magnitude
    effective_impact: float  # After system processing
    tension_target: str  # Which tension is primarily affected
    description: str = ""
    timestamp: int = 0

    # Abundance paradox indicators (for unicorns)
    causes_dutch_disease: bool = False
    causes_bubble: bool = False
    causes_institutional_capture: bool = False


@dataclass
class EventParameters:
    """Parameters for the event generation process."""

    # Base Poisson rate
    lambda_0: float = 0.01  # ~1 event per 100 periods

    # Tension sensitivity
    kappa: float = 2.0  # Sensitivity to tension in rate
    t_crit: float = 0.7  # Critical tension threshold

    # Black Swan parameters
    beta_amplification: float = 1.5
    alpha_threshold: float = 2.0
    xi_0: float = -0.05  # Critical amplification threshold

    # Unicorn parameters
    kappa_abs_0: float = 0.05  # Base absorption capacity
    gamma_memory: float = 0.3  # Memory contribution
    sigma_abundance: float = 0.02
    phi_threshold: float = 2.5  # "Too much success" threshold

    # Secondary effects
    omega_0: float = 0.01
    omega_1: float = 0.02
    omega_2: float = 0.03


class ExtremeEventGenerator:
    """
    Generator for extreme economic events.

    Implements the non-homogeneous Poisson process from Equation (10)
    and impact functions from Equations (11) and (12).
    """

    def __init__(
        self,
        params: Optional[EventParameters] = None,
        seed: Optional[int] = None,
    ):
        self.params = params or EventParameters()
        self.rng = np.random.default_rng(seed)

        # Event history
        self.event_history: List[ExtremeEvent] = []

        # Historical events for calibration reference
        self.historical_events = self._load_historical_events()

    def _load_historical_events(self) -> Dict[str, Dict[str, Any]]:
        """Load historical event parameters from Table 7."""
        return {
            "dotcom_crisis": {
                "year": 2000,
                "type": EventType.UNICORN,  # Bubble burst
                "t_f": 0.7,
                "magnitude": 0.15,  # Initial positive then crash
                "description": "Dot-com bubble correction",
            },
            "subprime_crisis": {
                "year": 2008,
                "type": EventType.BLACK_SWAN,
                "t_f": 0.9,
                "magnitude": -0.25,
                "description": "Subprime mortgage crisis",
            },
            "eurozone_crisis": {
                "year": 2010,
                "type": EventType.GREY_RHINO,
                "t_d": 0.8,
                "t_c": 0.6,
                "magnitude": -0.08,
                "description": "European sovereign debt crisis",
            },
            "covid_pandemic": {
                "year": 2020,
                "type": EventType.BLACK_SWAN,
                "t_c": 0.9,
                "magnitude": -0.15,
                "description": "COVID-19 global pandemic",
            },
            "post_covid_inflation": {
                "year": 2022,
                "type": EventType.GREY_RHINO,
                "t_e": 0.7,
                "t_x": 0.5,
                "magnitude": -0.05,
                "description": "Post-pandemic inflation surge",
            },
            "ukraine_war": {
                "year": 2022,
                "type": EventType.BLACK_SWAN,
                "t_e": 0.8,
                "magnitude": -0.08,
                "description": "Ukraine conflict energy shock",
            },
        }

    def compute_event_rate(
        self,
        t_adjusted: float,
    ) -> float:
        """
        Compute the instantaneous event rate λ(t).

        λ(t) = λ_0 * [1 + κ * tanh(T_adjusted(t) / T_crit)]

        Higher tension increases the probability of extreme events.
        """
        tension_factor = np.tanh(t_adjusted / self.params.t_crit)
        rate = self.params.lambda_0 * (1 + self.params.kappa * tension_factor)
        return max(0, rate)

    def generate_event(
        self,
        t_adjusted: float,
        macro_memory: float = 0.0,
        coupling: float = 0.5,
        sectoral_coherence: float = 0.7,
        financial_tension: float = 0.3,
    ) -> Optional[ExtremeEvent]:
        """
        Potentially generate an extreme event.

        Uses Poisson process with tension-dependent rate.
        """
        # Compute rate
        rate = self.compute_event_rate(t_adjusted)

        # Sample from Poisson (for discrete time, use Bernoulli approximation)
        if self.rng.random() > rate:
            return None  # No event

        # Generate event type and magnitude
        # Higher tensions favor black swans
        p_negative = 0.3 + 0.4 * t_adjusted
        is_negative = self.rng.random() < p_negative

        if is_negative:
            event = self._generate_black_swan(
                t_adjusted, macro_memory, financial_tension
            )
        else:
            event = self._generate_unicorn(
                t_adjusted, macro_memory, coupling, sectoral_coherence
            )

        event.timestamp = len(self.event_history)
        self.event_history.append(event)

        return event

    def _generate_black_swan(
        self,
        t_adjusted: float,
        macro_memory: float,
        financial_tension: float,
    ) -> ExtremeEvent:
        """
        Generate a Black Swan (negative) event.

        Impact_neg(ξ, t) = ξ * [1 + β * T_adj / (1 + exp(-α(ξ - ξ_0)))]
        """
        # Raw magnitude (negative)
        xi = -self.rng.exponential(0.05)  # Exponential for heavy tails
        xi = max(-0.5, xi)  # Cap at -50% shock

        # Compute amplification from Equation (11)
        threshold_effect = 1 / (
            1 + np.exp(-self.params.alpha_threshold * (xi - self.params.xi_0))
        )
        amplification = 1 + self.params.beta_amplification * t_adjusted * threshold_effect

        effective_impact = xi * amplification

        # Determine primary tension affected
        if financial_tension > 0.5:
            tension_target = "financial"
        elif t_adjusted > 0.6:
            tension_target = "events"
        else:
            tension_target = self.rng.choice(["energy", "trade", "currency"])

        return ExtremeEvent(
            event_type=EventType.BLACK_SWAN,
            magnitude=xi,
            effective_impact=effective_impact,
            tension_target=tension_target,
            description=f"Negative shock: {xi:.2%} raw, {effective_impact:.2%} effective",
        )

    def _generate_unicorn(
        self,
        t_adjusted: float,
        macro_memory: float,
        coupling: float,
        sectoral_coherence: float,
    ) -> ExtremeEvent:
        """
        Generate a Unicorn (positive) event with abundance paradox.

        Impact_pos(ξ, t) = ξ * exp(-(ξ - κ_abs)²/(2σ²)) - Ω * I{ξ > φ*κ_abs}
        """
        # Raw magnitude (positive)
        xi = self.rng.exponential(0.05)
        xi = min(0.3, xi)  # Cap at +30% shock

        # Compute absorption capacity from Equation (13)
        kappa_abs = (
            self.params.kappa_abs_0
            + self.params.gamma_memory * macro_memory * coupling
        )

        # Abundance penalty from Equation (14)
        omega = (
            self.params.omega_0
            + self.params.omega_1 * t_adjusted  # Financial tension contribution
            + self.params.omega_2 * (1 - sectoral_coherence)  # Incoherence contribution
        )

        # Effective impact with absorption and abundance paradox
        absorption_term = np.exp(
            -((xi - kappa_abs) ** 2) / (2 * self.params.sigma_abundance ** 2)
        )

        # Check for "too much success"
        abundance_threshold = self.params.phi_threshold * kappa_abs
        abundance_penalty = omega if xi > abundance_threshold else 0

        effective_impact = xi * absorption_term - abundance_penalty

        # Determine abundance paradox effects
        causes_dutch_disease = xi > 2 * kappa_abs and coupling > 0.7
        causes_bubble = xi > 1.5 * kappa_abs and sectoral_coherence < 0.5
        causes_institutional_capture = xi > 3 * kappa_abs and macro_memory < 0.3

        # Primary tension affected (usually reduced, but may trigger secondary)
        if causes_dutch_disease:
            tension_target = "trade"  # Dutch disease affects competitiveness
        elif causes_bubble:
            tension_target = "financial"  # Bubbles increase financial tension
        else:
            tension_target = "events"  # General positive shock

        return ExtremeEvent(
            event_type=EventType.UNICORN,
            magnitude=xi,
            effective_impact=effective_impact,
            tension_target=tension_target,
            description=f"Positive shock: {xi:.2%} raw, {effective_impact:.2%} effective",
            causes_dutch_disease=causes_dutch_disease,
            causes_bubble=causes_bubble,
            causes_institutional_capture=causes_institutional_capture,
        )

    def inject_historical_event(
        self,
        event_name: str,
        current_period: int,
    ) -> Optional[ExtremeEvent]:
        """
        Inject a historical event for calibration purposes.

        Useful for validating model against known shocks.
        """
        if event_name not in self.historical_events:
            return None

        event_data = self.historical_events[event_name]

        event = ExtremeEvent(
            event_type=event_data["type"],
            magnitude=event_data["magnitude"],
            effective_impact=event_data["magnitude"],  # Will be recomputed
            tension_target="events",
            description=event_data["description"],
            timestamp=current_period,
        )

        self.event_history.append(event)
        return event

    def compute_systemic_impact(
        self,
        event: ExtremeEvent,
        network_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute how an event propagates through the system.

        Considers network topology effects:
        - Hub infection accelerates propagation
        - Scale-free fragility to targeted shocks
        """
        impact = {"direct": event.effective_impact}

        # Network amplification
        avg_degree = network_metrics.get("average_degree", 5)
        max_degree = network_metrics.get("max_degree", 50)
        clustering = network_metrics.get("clustering", 0.3)

        # Direct impact spreads through network
        diffusion_factor = np.log(avg_degree + 1) / np.log(10)

        # Hub concentration amplifies shocks
        hub_factor = 1 + 0.2 * (max_degree / avg_degree - 1)

        # Clustering provides local resilience
        resilience = 1 - 0.3 * clustering

        # Total propagation
        propagated_impact = (
            event.effective_impact * diffusion_factor * hub_factor * resilience
        )

        impact["propagated"] = propagated_impact
        impact["total"] = event.effective_impact + propagated_impact * 0.5

        # Sector-specific impacts
        if event.event_type == EventType.BLACK_SWAN:
            impact["financial_sector"] = impact["total"] * 1.2
            impact["real_sector"] = impact["total"] * 0.8
        else:
            impact["financial_sector"] = impact["total"] * 0.8
            impact["real_sector"] = impact["total"] * 1.1

        return impact

    def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about historical events."""
        if not self.event_history:
            return {"count": 0}

        black_swans = [
            e for e in self.event_history if e.event_type == EventType.BLACK_SWAN
        ]
        unicorns = [
            e for e in self.event_history if e.event_type == EventType.UNICORN
        ]

        return {
            "count": len(self.event_history),
            "black_swan_count": len(black_swans),
            "unicorn_count": len(unicorns),
            "avg_magnitude": np.mean([abs(e.magnitude) for e in self.event_history]),
            "avg_impact": np.mean([e.effective_impact for e in self.event_history]),
            "dutch_disease_events": sum(1 for e in unicorns if e.causes_dutch_disease),
            "bubble_events": sum(1 for e in unicorns if e.causes_bubble),
        }

    def reset(self) -> None:
        """Reset event history."""
        self.event_history.clear()
