"""
GDP Directional Vector

Implements the GDP vector from Section 4.3:
v_PIB(t) = (g_t, a_t, theta_t)

Where:
- g_t: Instantaneous growth rate = (Y_t - Y_{t-1}) / Y_{t-1}
- a_t: Acceleration/deceleration = (g_t - g_{t-1}) / dt
- theta_t: Sectoral coherence = avg correlation of sector growth with total
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class GDPVector:
    """
    GDP directional vector providing richer information than scalar GDP.

    Components:
    - g: Growth rate (instantaneous)
    - a: Acceleration (second derivative)
    - theta: Sectoral coherence (correlation measure)
    """

    g: float = 0.02  # Growth rate
    a: float = 0.0  # Acceleration
    theta: float = 0.7  # Sectoral coherence

    # Historical values for computation
    gdp_history: List[float] = field(default_factory=list)
    growth_history: List[float] = field(default_factory=list)
    sector_growth_history: Dict[str, List[float]] = field(default_factory=dict)

    # Configuration
    coherence_window: int = 4  # Quarters for correlation calculation
    sectors: List[str] = field(
        default_factory=lambda: [
            "primary",
            "manufacturing",
            "services",
            "financial",
            "public",
        ]
    )

    def update(
        self,
        current_gdp: float,
        sector_outputs: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Update the GDP vector with new period data.

        Args:
            current_gdp: Total GDP this period
            sector_outputs: Output by sector (optional)

        Returns:
            Tuple of (g_t, a_t, theta_t)
        """
        # Store GDP
        self.gdp_history.append(current_gdp)

        # Compute growth rate
        if len(self.gdp_history) >= 2:
            prev_gdp = self.gdp_history[-2]
            if prev_gdp > 0:
                self.g = (current_gdp - prev_gdp) / prev_gdp
            else:
                self.g = 0.0
        else:
            self.g = 0.02  # Default initial growth

        # Store growth
        self.growth_history.append(self.g)

        # Compute acceleration
        if len(self.growth_history) >= 2:
            self.a = self.growth_history[-1] - self.growth_history[-2]
        else:
            self.a = 0.0

        # Update sectoral data and compute coherence
        if sector_outputs:
            self.theta = self._compute_sectoral_coherence(sector_outputs)
        else:
            # Default coherence based on phase characteristics
            if self.g > 0.02 and self.a > 0:
                self.theta = 0.7 + 0.1 * min(1, self.g / 0.05)
            elif self.g < 0:
                self.theta = 0.3 + 0.2 * max(-1, self.g / 0.05)
            else:
                self.theta = 0.6

        return self.g, self.a, self.theta

    def _compute_sectoral_coherence(
        self,
        sector_outputs: Dict[str, float],
    ) -> float:
        """
        Compute sectoral coherence theta_t.

        theta_t = (1/N) * sum_s corr(g_s, g_total)

        Measures how synchronized sector growth is with aggregate growth.
        """
        # Update sector histories
        for sector in self.sectors:
            if sector not in self.sector_growth_history:
                self.sector_growth_history[sector] = []

            if sector in sector_outputs:
                current = sector_outputs[sector]
                history = self.sector_growth_history[sector]

                if history:
                    prev = history[-1] if history else current
                    if prev > 0:
                        sector_growth = (current - prev) / prev
                    else:
                        sector_growth = 0.0
                else:
                    sector_growth = self.g  # Use aggregate as default

                self.sector_growth_history[sector].append(sector_growth)

        # Compute correlations if enough history
        if len(self.growth_history) < self.coherence_window:
            return 0.7  # Default

        total_growth = np.array(self.growth_history[-self.coherence_window:])

        correlations = []
        for sector in self.sectors:
            if sector in self.sector_growth_history:
                sector_growth = self.sector_growth_history[sector]
                if len(sector_growth) >= self.coherence_window:
                    sector_arr = np.array(sector_growth[-self.coherence_window:])

                    # Compute correlation
                    if np.std(total_growth) > 0 and np.std(sector_arr) > 0:
                        corr = np.corrcoef(total_growth, sector_arr)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

        if correlations:
            return np.mean(correlations)
        return 0.7

    def get_phase_indicators(self) -> Dict[str, str]:
        """
        Get qualitative indicators based on vector values.

        Uses Table 2 interpretations.
        """
        indicators = {}

        # Growth direction
        if self.g > 0.05:
            indicators["growth"] = "rapid"
        elif self.g > 0.02:
            indicators["growth"] = "moderate"
        elif self.g > 0:
            indicators["growth"] = "slow"
        elif self.g > -0.03:
            indicators["growth"] = "contracting"
        else:
            indicators["growth"] = "severe_contraction"

        # Momentum
        if self.a > 0.01:
            indicators["momentum"] = "accelerating"
        elif self.a > -0.01:
            indicators["momentum"] = "stable"
        else:
            indicators["momentum"] = "decelerating"

        # Coordination
        if self.theta > 0.8:
            indicators["coordination"] = "highly_coordinated"
        elif self.theta > 0.6:
            indicators["coordination"] = "coordinated"
        elif self.theta > 0.4:
            indicators["coordination"] = "mixed"
        else:
            indicators["coordination"] = "fragmented"

        return indicators

    def estimate_phase(self) -> str:
        """
        Estimate the likely economic phase from vector values.

        Based on Table 2 ranges.
        """
        if self.g > 0.05 and self.a < 0 and self.theta < 0.7:
            return "overheating"
        elif self.g < 0 and self.a < 0:
            return "crisis"
        elif self.g < 0 and self.a > 0:
            return "recession"
        elif 0 < self.g <= 0.02 and self.a > 0:
            return "activation"
        elif 0.02 < self.g <= 0.05 and self.a > 0 and self.theta > 0.6:
            return "expansion"
        elif 0.02 < self.g <= 0.04 and abs(self.a) < 0.01 and self.theta > 0.7:
            return "maturity"
        else:
            return "transition"

    def get_momentum_signal(self) -> float:
        """
        Get a momentum signal combining growth and acceleration.

        Positive = bullish, Negative = bearish
        Range approximately [-1, 1]
        """
        # Normalize components
        g_norm = np.tanh(self.g * 20)  # Scale growth
        a_norm = np.tanh(self.a * 100)  # Scale acceleration

        # Weight acceleration more during turning points
        if np.sign(self.g) != np.sign(self.a):
            # Divergence - potential turning point
            return 0.3 * g_norm + 0.7 * a_norm
        else:
            # Convergence - trend confirmation
            return 0.7 * g_norm + 0.3 * a_norm

    def get_stability_index(self) -> float:
        """
        Compute a stability index based on recent volatility.

        Higher = more stable (0 to 1)
        """
        if len(self.growth_history) < 4:
            return 0.7

        recent_growth = self.growth_history[-8:]
        volatility = np.std(recent_growth) if len(recent_growth) > 1 else 0

        # Combine with coherence
        stability = self.theta * 0.5 + 0.5 * np.exp(-volatility * 10)

        return np.clip(stability, 0, 1)

    def project_growth(self, horizons: List[int] = None) -> Dict[int, float]:
        """
        Simple projection of growth rates using AR(2) approximation.

        Args:
            horizons: List of quarters ahead to project

        Returns:
            Dict mapping horizon to projected growth
        """
        if horizons is None:
            horizons = [1, 2, 4]

        if len(self.growth_history) < 3:
            return {h: self.g for h in horizons}

        # Simple AR(2): g_{t+1} = c + phi1*g_t + phi2*g_{t-1}
        recent = self.growth_history[-10:]
        if len(recent) >= 3:
            mean_g = np.mean(recent)
            ar1_coef = 0.6
            ar2_coef = 0.2

            projections = {}
            g_prev = self.growth_history[-2]
            g_curr = self.g

            for h in horizons:
                for _ in range(h):
                    g_next = (
                        mean_g * (1 - ar1_coef - ar2_coef)
                        + ar1_coef * g_curr
                        + ar2_coef * g_prev
                    )
                    g_prev = g_curr
                    g_curr = g_next

                projections[h] = g_curr

            return projections

        return {h: self.g for h in horizons}

    def to_dict(self) -> Dict[str, float]:
        """Export current vector values."""
        return {
            "g": self.g,
            "a": self.a,
            "theta": self.theta,
            "momentum": self.get_momentum_signal(),
            "stability": self.get_stability_index(),
        }

    def reset(self) -> None:
        """Reset all history."""
        self.gdp_history.clear()
        self.growth_history.clear()
        self.sector_growth_history.clear()
        self.g = 0.02
        self.a = 0.0
        self.theta = 0.7
