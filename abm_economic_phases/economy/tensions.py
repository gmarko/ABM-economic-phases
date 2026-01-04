"""
Systemic Tension Indices

Implements the tension system from Section 5:
- T_adjusted(t) = sum_i(w_i(t) * T_i(t)) / (1 + lambda * M_macro(t))
- Five tension dimensions: Energy, Trade, Currency, Financial, Events

Adaptive weights evolve according to Equation (10):
w_i(t+1) = w_i(t) + eta * (dT_adj/dT_i * Historical_Impact_i)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TensionMetrics:
    """Individual tension component metrics."""

    # Raw values
    t_energy: float = 0.0  # T_E: Energy dependence × volatility
    t_trade: float = 0.0  # T_C: Trade restrictions × concentration
    t_currency: float = 0.0  # T_D: FX volatility × external exposure
    t_financial: float = 0.0  # T_F: Spreads + leverage × credit growth
    t_events: float = 0.0  # T_X: Event frequency × surprise × impact

    # Adjusted aggregate
    t_adjusted: float = 0.0

    # Weights (adaptive)
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "energy": 0.2,
            "trade": 0.2,
            "currency": 0.2,
            "financial": 0.25,
            "events": 0.15,
        }
    )


class TensionSystem:
    """
    System for computing and updating structural tensions.

    Implements operational definitions from Table 3:
    - T_E: (Energy imports / GDP) × Volatility_30d(oil price)
    - T_C: Restriction index × (1 - A_t) × Export concentration
    - T_D: Volatility_60d(RER) × External exposure
    - T_F: Corporate spread + Aggregate leverage × Credit growth
    - T_X: Event frequency × Informational surprise × Initial impact
    """

    def __init__(
        self,
        memory_decay: float = 0.1,  # Lambda for memory adjustment
        learning_rate: float = 0.05,  # Eta for weight adaptation
        seed: Optional[int] = None,
    ):
        self.memory_decay = memory_decay
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # Current metrics
        self.metrics = TensionMetrics()

        # History for tracking
        self.tension_history: List[TensionMetrics] = []
        self.impact_history: Dict[str, List[float]] = {
            "energy": [],
            "trade": [],
            "currency": [],
            "financial": [],
            "events": [],
        }

    def compute_energy_tension(
        self,
        energy_import_gdp_ratio: float = 0.05,
        oil_volatility_30d: float = 0.1,
        renewable_share: float = 0.2,
    ) -> float:
        """
        Compute energy tension T_E.

        T_E = (Energy imports / GDP) × Volatility_30d(oil) × (1 - renewable_share)
        """
        t_e = energy_import_gdp_ratio * oil_volatility_30d * (1 - renewable_share)
        return np.clip(t_e, 0, 1)

    def compute_trade_tension(
        self,
        restriction_index: float = 0.3,
        coupling: float = 0.5,
        export_concentration: float = 0.4,
    ) -> float:
        """
        Compute trade tension T_C.

        T_C = Restriction index × (1 - A_t) × Export concentration
        """
        t_c = restriction_index * (1 - coupling) * export_concentration
        return np.clip(t_c, 0, 1)

    def compute_currency_tension(
        self,
        fx_volatility_60d: float = 0.05,
        external_exposure: float = 0.3,
        reserves_months: float = 6.0,
    ) -> float:
        """
        Compute currency tension T_D.

        T_D = Volatility_60d(RER) × External exposure × (1 - reserve_buffer)
        """
        reserve_buffer = min(1, reserves_months / 12)  # Normalize to 1 year
        t_d = fx_volatility_60d * external_exposure * (1 - reserve_buffer * 0.5)
        return np.clip(t_d, 0, 1)

    def compute_financial_tension(
        self,
        corporate_spread: float = 0.02,
        aggregate_leverage: float = 0.5,
        credit_growth: float = 0.05,
        npl_ratio: float = 0.02,
    ) -> float:
        """
        Compute financial tension T_F.

        T_F = Corporate spread + Leverage × Credit growth + NPL penalty
        """
        # Normalize spread (typical range 0.01-0.05)
        spread_norm = corporate_spread / 0.05

        # Credit acceleration term
        credit_term = aggregate_leverage * max(0, credit_growth - 0.02)

        # NPL penalty
        npl_penalty = npl_ratio / 0.05

        t_f = spread_norm + credit_term + npl_penalty
        return np.clip(t_f, 0, 1)

    def compute_event_tension(
        self,
        event_frequency: float = 0.0,
        surprise_index: float = 0.0,
        initial_impact: float = 0.0,
    ) -> float:
        """
        Compute event tension T_X.

        T_X = Event frequency × Informational surprise × Initial impact
        """
        t_x = event_frequency * surprise_index * abs(initial_impact)
        return np.clip(t_x, 0, 1)

    def update(
        self,
        macro_memory: float = 0.0,
        energy_inputs: Optional[Dict] = None,
        trade_inputs: Optional[Dict] = None,
        currency_inputs: Optional[Dict] = None,
        financial_inputs: Optional[Dict] = None,
        event_inputs: Optional[Dict] = None,
    ) -> TensionMetrics:
        """
        Update all tension metrics.

        Args:
            macro_memory: M_macro(t) for adjustment
            *_inputs: Dicts with specific inputs for each tension

        Returns:
            Updated TensionMetrics
        """
        # Compute individual tensions
        if energy_inputs:
            self.metrics.t_energy = self.compute_energy_tension(**energy_inputs)
        else:
            # Random walk with mean reversion
            self.metrics.t_energy += self.rng.normal(0, 0.02)
            self.metrics.t_energy = np.clip(
                0.9 * self.metrics.t_energy + 0.1 * 0.3, 0, 1
            )

        if trade_inputs:
            self.metrics.t_trade = self.compute_trade_tension(**trade_inputs)
        else:
            self.metrics.t_trade += self.rng.normal(0, 0.015)
            self.metrics.t_trade = np.clip(
                0.9 * self.metrics.t_trade + 0.1 * 0.25, 0, 1
            )

        if currency_inputs:
            self.metrics.t_currency = self.compute_currency_tension(**currency_inputs)
        else:
            self.metrics.t_currency += self.rng.normal(0, 0.02)
            self.metrics.t_currency = np.clip(
                0.9 * self.metrics.t_currency + 0.1 * 0.2, 0, 1
            )

        if financial_inputs:
            self.metrics.t_financial = self.compute_financial_tension(**financial_inputs)
        else:
            self.metrics.t_financial += self.rng.normal(0, 0.025)
            self.metrics.t_financial = np.clip(
                0.9 * self.metrics.t_financial + 0.1 * 0.25, 0, 1
            )

        if event_inputs:
            self.metrics.t_events = self.compute_event_tension(**event_inputs)
        else:
            # Events are sparse, tend toward zero
            self.metrics.t_events *= 0.5
            self.metrics.t_events = max(0, self.metrics.t_events)

        # Compute adjusted tension using Equation (9)
        self._compute_adjusted_tension(macro_memory)

        # Store history
        self.tension_history.append(
            TensionMetrics(
                t_energy=self.metrics.t_energy,
                t_trade=self.metrics.t_trade,
                t_currency=self.metrics.t_currency,
                t_financial=self.metrics.t_financial,
                t_events=self.metrics.t_events,
                t_adjusted=self.metrics.t_adjusted,
                weights=self.metrics.weights.copy(),
            )
        )

        return self.metrics

    def _compute_adjusted_tension(self, macro_memory: float) -> None:
        """
        Compute adjusted tension using Equation (9):
        T_adjusted(t) = sum_i(w_i(t) * T_i(t)) / (1 + lambda * M_macro(t))
        """
        tensions = {
            "energy": self.metrics.t_energy,
            "trade": self.metrics.t_trade,
            "currency": self.metrics.t_currency,
            "financial": self.metrics.t_financial,
            "events": self.metrics.t_events,
        }

        # Weighted sum
        weighted_sum = sum(
            self.metrics.weights[key] * tensions[key] for key in tensions
        )

        # Memory adjustment (higher memory = lower effective tension)
        adjustment = 1 + self.memory_decay * macro_memory

        self.metrics.t_adjusted = weighted_sum / adjustment

    def update_weights(
        self,
        gdp_impact: float,
        dominant_tension: Optional[str] = None,
    ) -> None:
        """
        Update adaptive weights based on observed impacts.

        Equation (10): w_i(t+1) = w_i(t) + eta * (dT_adj/dT_i * Impact_i)
        """
        if dominant_tension and dominant_tension in self.metrics.weights:
            # Increase weight of the tension that had impact
            old_weight = self.metrics.weights[dominant_tension]
            new_weight = old_weight + self.learning_rate * abs(gdp_impact)

            self.metrics.weights[dominant_tension] = min(0.5, new_weight)

            # Renormalize weights
            total = sum(self.metrics.weights.values())
            self.metrics.weights = {
                k: v / total for k, v in self.metrics.weights.items()
            }

        # Track impact history
        if dominant_tension:
            self.impact_history[dominant_tension].append(gdp_impact)

    def get_tension_vector(self) -> np.ndarray:
        """Get tensions as numpy array [T_E, T_C, T_D, T_F, T_X]."""
        return np.array([
            self.metrics.t_energy,
            self.metrics.t_trade,
            self.metrics.t_currency,
            self.metrics.t_financial,
            self.metrics.t_events,
        ])

    def get_dominant_tension(self) -> Tuple[str, float]:
        """Identify the currently dominant tension source."""
        tensions = {
            "energy": self.metrics.t_energy,
            "trade": self.metrics.t_trade,
            "currency": self.metrics.t_currency,
            "financial": self.metrics.t_financial,
            "events": self.metrics.t_events,
        }

        dominant = max(tensions, key=tensions.get)
        return dominant, tensions[dominant]

    def get_risk_assessment(self) -> Dict[str, str]:
        """Get qualitative risk assessment for each tension."""
        def assess(value: float) -> str:
            if value < 0.2:
                return "low"
            elif value < 0.4:
                return "moderate"
            elif value < 0.6:
                return "elevated"
            elif value < 0.8:
                return "high"
            else:
                return "extreme"

        return {
            "energy": assess(self.metrics.t_energy),
            "trade": assess(self.metrics.t_trade),
            "currency": assess(self.metrics.t_currency),
            "financial": assess(self.metrics.t_financial),
            "events": assess(self.metrics.t_events),
            "overall": assess(self.metrics.t_adjusted),
        }

    def simulate_shock(
        self,
        shock_type: str,
        magnitude: float,
    ) -> float:
        """
        Simulate a tension shock and return the adjusted tension impact.

        Args:
            shock_type: One of 'energy', 'trade', 'currency', 'financial', 'events'
            magnitude: Shock magnitude (0-1 scale)

        Returns:
            New adjusted tension level
        """
        magnitude = np.clip(magnitude, 0, 1)

        if shock_type == "energy":
            self.metrics.t_energy = min(1, self.metrics.t_energy + magnitude)
        elif shock_type == "trade":
            self.metrics.t_trade = min(1, self.metrics.t_trade + magnitude)
        elif shock_type == "currency":
            self.metrics.t_currency = min(1, self.metrics.t_currency + magnitude)
        elif shock_type == "financial":
            self.metrics.t_financial = min(1, self.metrics.t_financial + magnitude)
        elif shock_type == "events":
            self.metrics.t_events = min(1, self.metrics.t_events + magnitude)

        # Recompute adjusted
        self._compute_adjusted_tension(0.0)

        return self.metrics.t_adjusted

    def to_dict(self) -> Dict[str, float]:
        """Export current tensions as dictionary."""
        return {
            "t_energy": self.metrics.t_energy,
            "t_trade": self.metrics.t_trade,
            "t_currency": self.metrics.t_currency,
            "t_financial": self.metrics.t_financial,
            "t_events": self.metrics.t_events,
            "t_adjusted": self.metrics.t_adjusted,
            **{f"weight_{k}": v for k, v in self.metrics.weights.items()},
        }

    def reset(self) -> None:
        """Reset tensions to initial state."""
        self.metrics = TensionMetrics()
        self.tension_history.clear()
        for key in self.impact_history:
            self.impact_history[key].clear()
