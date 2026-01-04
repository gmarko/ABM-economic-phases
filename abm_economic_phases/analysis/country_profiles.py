"""
Country-Specific Analysis Module

Implements Section 6.3 of the paper: Comparative Country Analysis

Provides calibration profiles for different economies with distinct characteristics:
- Spain: Synchronized expansion, high unemployment sensitivity
- Germany: Export-driven, vulnerable to energy shocks
- France: Stagnant stability, rigid labor markets
- UK: Post-Brexit permanent structural fracture
- USA: Cyclical crises with strong recovery capacity
- China: High growth masking low coherence and structural fragility

Each profile includes:
- Initial tension weights
- Phase sensitivity parameters
- Network topology characteristics
- MMT policy space constraints
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CountryProfile:
    """Calibration profile for a specific country/economy."""

    name: str
    code: str  # ISO 3166-1 alpha-2

    # Structural parameters
    initial_gdp_growth: float = 0.02
    potential_output_growth: float = 0.02
    natural_unemployment: float = 0.05

    # Tension weights (initial)
    tension_weights: Dict[str, float] = field(default_factory=lambda: {
        "energy": 0.2, "trade": 0.2, "currency": 0.2,
        "financial": 0.25, "events": 0.15
    })

    # External coupling
    coupling: float = 0.5  # A_t: 0 = isolated, 1 = fully integrated

    # Network characteristics
    avg_network_degree: float = 6.0
    hub_concentration: float = 0.3  # Degree of scale-free concentration

    # Phase transition sensitivities
    crisis_threshold: float = 0.8  # T_adj threshold for crisis
    recovery_speed: float = 1.0  # Multiplier on recovery rate

    # MMT policy constraints
    fiscal_space: str = "full"  # "full", "constrained", "eurozone"
    monetary_sovereignty: bool = True

    # Coherence characteristics
    sectoral_diversity: float = 0.7  # Higher = more sectors
    base_coherence: float = 0.6

    # Historical events sensitivity
    event_amplification: float = 1.0


# Pre-defined country profiles from Section 6.3

SPAIN = CountryProfile(
    name="Spain",
    code="ES",
    initial_gdp_growth=0.025,
    potential_output_growth=0.02,
    natural_unemployment=0.12,  # Higher structural unemployment
    tension_weights={
        "energy": 0.25,  # Energy import dependence
        "trade": 0.15,
        "currency": 0.1,  # Euro member
        "financial": 0.35,  # Real estate sensitivity
        "events": 0.15,
    },
    coupling=0.7,  # Highly integrated in EU
    avg_network_degree=5.0,
    hub_concentration=0.35,
    crisis_threshold=0.75,
    recovery_speed=0.8,  # Slower recovery
    fiscal_space="eurozone",  # SGP constraints
    monetary_sovereignty=False,  # Euro member
    sectoral_diversity=0.6,
    base_coherence=0.65,
    event_amplification=1.2,  # More vulnerable to shocks
)

GERMANY = CountryProfile(
    name="Germany",
    code="DE",
    initial_gdp_growth=0.015,
    potential_output_growth=0.012,
    natural_unemployment=0.04,
    tension_weights={
        "energy": 0.35,  # Gas dependence (especially post-2022)
        "trade": 0.25,  # Export-driven
        "currency": 0.05,  # Euro member
        "financial": 0.2,
        "events": 0.15,
    },
    coupling=0.8,  # Core of EU integration
    avg_network_degree=8.0,  # Dense industrial network
    hub_concentration=0.4,  # Strong Mittelstand + DAX
    crisis_threshold=0.85,  # More resilient
    recovery_speed=1.2,
    fiscal_space="eurozone",
    monetary_sovereignty=False,
    sectoral_diversity=0.8,  # Highly diversified
    base_coherence=0.75,
    event_amplification=0.9,
)

FRANCE = CountryProfile(
    name="France",
    code="FR",
    initial_gdp_growth=0.012,
    potential_output_growth=0.01,  # Lower potential
    natural_unemployment=0.08,
    tension_weights={
        "energy": 0.15,  # Nuclear power
        "trade": 0.2,
        "currency": 0.1,
        "financial": 0.3,
        "events": 0.25,  # Social unrest sensitivity
    },
    coupling=0.65,
    avg_network_degree=6.0,
    hub_concentration=0.45,  # Paris concentration
    crisis_threshold=0.8,
    recovery_speed=0.9,
    fiscal_space="eurozone",
    monetary_sovereignty=False,
    sectoral_diversity=0.7,
    base_coherence=0.55,  # Lower coherence
    event_amplification=1.1,
)

UK = CountryProfile(
    name="United Kingdom",
    code="GB",
    initial_gdp_growth=0.015,
    potential_output_growth=0.012,
    natural_unemployment=0.045,
    tension_weights={
        "energy": 0.2,
        "trade": 0.35,  # Brexit impact
        "currency": 0.15,  # GBP volatility
        "financial": 0.2,  # City of London
        "events": 0.1,
    },
    coupling=0.4,  # Reduced post-Brexit
    avg_network_degree=7.0,
    hub_concentration=0.5,  # London concentration
    crisis_threshold=0.75,
    recovery_speed=1.0,
    fiscal_space="full",  # Sovereign currency
    monetary_sovereignty=True,
    sectoral_diversity=0.65,
    base_coherence=0.5,  # Structural fracture
    event_amplification=1.3,  # Brexit amplification
)

USA = CountryProfile(
    name="United States",
    code="US",
    initial_gdp_growth=0.025,
    potential_output_growth=0.02,
    natural_unemployment=0.04,
    tension_weights={
        "energy": 0.15,  # Energy independence
        "trade": 0.2,
        "currency": 0.05,  # Dollar privilege
        "financial": 0.4,  # Financial markets central
        "events": 0.2,
    },
    coupling=0.3,  # Large domestic market
    avg_network_degree=10.0,  # Dense, complex economy
    hub_concentration=0.35,
    crisis_threshold=0.85,
    recovery_speed=1.5,  # Strong recovery capacity
    fiscal_space="full",
    monetary_sovereignty=True,
    sectoral_diversity=0.9,
    base_coherence=0.7,
    event_amplification=0.8,  # Resilient
)

CHINA = CountryProfile(
    name="China",
    code="CN",
    initial_gdp_growth=0.055,
    potential_output_growth=0.045,
    natural_unemployment=0.04,
    tension_weights={
        "energy": 0.2,
        "trade": 0.25,  # Export dependence
        "currency": 0.1,  # Managed float
        "financial": 0.3,  # Shadow banking, real estate
        "events": 0.15,
    },
    coupling=0.5,  # Selective integration
    avg_network_degree=6.0,
    hub_concentration=0.5,  # SOE concentration
    crisis_threshold=0.7,  # Lower threshold (fragility)
    recovery_speed=1.3,  # State-driven recovery
    fiscal_space="full",
    monetary_sovereignty=True,
    sectoral_diversity=0.75,
    base_coherence=0.45,  # LOW coherence despite high growth
    event_amplification=1.0,
)

# Dictionary of all profiles
COUNTRY_PROFILES: Dict[str, CountryProfile] = {
    "ES": SPAIN,
    "DE": GERMANY,
    "FR": FRANCE,
    "GB": UK,
    "US": USA,
    "CN": CHINA,
}


class CountryComparator:
    """
    Comparative analysis tool for different country profiles.

    Implements the cross-country analysis from Section 6.3:
    - Phase synchronization analysis
    - Coherence vs growth divergence (early warning)
    - Cyclic vs structural crisis patterns
    """

    def __init__(self, countries: List[str] = None):
        """
        Initialize comparator with selected countries.

        Args:
            countries: List of country codes (default: all)
        """
        if countries is None:
            self.profiles = COUNTRY_PROFILES
        else:
            self.profiles = {c: COUNTRY_PROFILES[c] for c in countries
                            if c in COUNTRY_PROFILES}

    def compare_fragility(self) -> Dict[str, Dict[str, float]]:
        """
        Compare structural fragility indicators across countries.

        Returns fragility scores based on:
        - Crisis threshold (lower = more fragile)
        - Event amplification
        - Base coherence (lower = more fragile)
        - Recovery speed (lower = more persistent crises)
        """
        results = {}
        for code, profile in self.profiles.items():
            # Fragility index (0-1, higher = more fragile)
            fragility = (
                0.3 * (1 - profile.crisis_threshold)  # Lower threshold = fragile
                + 0.25 * (profile.event_amplification - 0.8) / 0.5  # Higher amp = fragile
                + 0.25 * (1 - profile.base_coherence)  # Lower coherence = fragile
                + 0.2 * (1 - profile.recovery_speed / 1.5)  # Slower recovery = fragile
            )

            results[code] = {
                "fragility_index": np.clip(fragility, 0, 1),
                "crisis_threshold": profile.crisis_threshold,
                "event_amplification": profile.event_amplification,
                "base_coherence": profile.base_coherence,
                "recovery_speed": profile.recovery_speed,
            }

        return results

    def compare_policy_space(self) -> Dict[str, Dict[str, any]]:
        """
        Compare policy space constraints across countries.

        Eurozone members have constrained fiscal space.
        Non-sovereign currency issuers cannot use full MMT toolkit.
        """
        results = {}
        for code, profile in self.profiles.items():
            # Policy flexibility score
            fiscal_score = {
                "full": 1.0,
                "constrained": 0.5,
                "eurozone": 0.3,
            }.get(profile.fiscal_space, 0.5)

            monetary_score = 1.0 if profile.monetary_sovereignty else 0.2

            results[code] = {
                "fiscal_space": profile.fiscal_space,
                "monetary_sovereignty": profile.monetary_sovereignty,
                "fiscal_score": fiscal_score,
                "monetary_score": monetary_score,
                "policy_flexibility": 0.6 * fiscal_score + 0.4 * monetary_score,
            }

        return results

    def identify_early_warning_divergence(
        self,
        gdp_growth: Dict[str, float],
        coherence: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Identify growth-coherence divergence (early warning signal).

        Per Section 6.4: Divergence between GDP growth and θ_t signals
        fragility before collapse (e.g., 2006-2007 pre-crisis).

        Args:
            gdp_growth: Current GDP growth by country
            coherence: Current sectoral coherence by country

        Returns:
            Warning indicators by country
        """
        results = {}
        for code in self.profiles:
            if code in gdp_growth and code in coherence:
                g = gdp_growth[code]
                theta = coherence[code]

                # Divergence: high growth with falling/low coherence
                # Normalize growth to 0-1 scale (0% to 5%)
                g_norm = np.clip(g / 0.05, 0, 1)

                # Warning if growth high but coherence low
                divergence = g_norm - theta
                warning_level = np.clip(divergence, 0, 1)

                results[code] = {
                    "gdp_growth": g,
                    "coherence": theta,
                    "divergence": divergence,
                    "warning_level": warning_level,
                    "interpretation": self._interpret_warning(warning_level),
                }

        return results

    def _interpret_warning(self, level: float) -> str:
        """Interpret warning level."""
        if level < 0.1:
            return "stable"
        elif level < 0.25:
            return "watch"
        elif level < 0.5:
            return "elevated"
        elif level < 0.75:
            return "high"
        else:
            return "critical"

    def compare_crisis_patterns(self) -> Dict[str, str]:
        """
        Classify countries by crisis pattern type.

        Types from Section 6.5:
        - Cyclical: Regular boom-bust cycles with recovery (USA)
        - Structural: Persistent fragility masked by growth (China)
        - Stagnant: Low volatility but also low growth (France)
        - Fractured: Permanent structural break (UK post-Brexit)
        """
        patterns = {}
        for code, profile in self.profiles.items():
            # Classify based on profile characteristics
            if profile.recovery_speed > 1.2 and profile.base_coherence > 0.6:
                pattern = "cyclical"  # Strong recovery, good coherence
            elif profile.base_coherence < 0.5 and profile.initial_gdp_growth > 0.03:
                pattern = "structural"  # High growth, low coherence
            elif profile.potential_output_growth < 0.015 and profile.recovery_speed < 1.0:
                pattern = "stagnant"  # Low potential, slow recovery
            elif profile.event_amplification > 1.2 and profile.base_coherence < 0.55:
                pattern = "fractured"  # High amplification, low coherence
            else:
                pattern = "mixed"

            patterns[code] = pattern

        return patterns

    def get_calibration_for_simulation(
        self,
        country_code: str,
    ) -> Dict[str, any]:
        """
        Get full calibration parameters for simulation.

        Returns all parameters needed to initialize the ABM for a specific country.
        """
        if country_code not in self.profiles:
            raise ValueError(f"Unknown country code: {country_code}")

        profile = self.profiles[country_code]

        return {
            "name": profile.name,
            "code": profile.code,
            # Initial conditions
            "initial_gdp_growth": profile.initial_gdp_growth,
            "potential_output_growth": profile.potential_output_growth,
            "natural_unemployment": profile.natural_unemployment,
            "base_coherence": profile.base_coherence,
            # Tensions
            "tension_weights": profile.tension_weights,
            "coupling": profile.coupling,
            # Network
            "avg_network_degree": profile.avg_network_degree,
            "hub_concentration": profile.hub_concentration,
            # Dynamics
            "crisis_threshold": profile.crisis_threshold,
            "recovery_speed": profile.recovery_speed,
            "event_amplification": profile.event_amplification,
            # Policy
            "fiscal_space": profile.fiscal_space,
            "monetary_sovereignty": profile.monetary_sovereignty,
            "sectoral_diversity": profile.sectoral_diversity,
        }
