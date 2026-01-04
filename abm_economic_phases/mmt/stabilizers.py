"""
MMT Compatibility and Stabilization Mechanisms

Implements Section 11:
- Operational budget constraint: G + i*D_{-1} = T + ΔD + ΔH
- Inflation as capacity phenomenon: π = β_0 + β_1*(Y/Y_pot) + β_2*T_E + β_3*T_C + β_4*E[π_{t+1}]
- Enhanced automatic stabilizer: G = G_bar - α*(Y-Y_pot) + δ*T_adj - γ*I_crisis
- Employer of Last Resort: L_ELR = max(0, L_target - L_private)

Key MMT insights incorporated:
1. Limits are real (resources, productivity), not purely financial
2. Inflation emerges from structural tensions and capacity constraints
3. Government acts as system stabilizer
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class FiscalState:
    """State of fiscal policy."""

    spending: float = 0.2  # G/GDP
    tax_revenue: float = 0.18  # T/GDP
    deficit: float = 0.02  # (G-T)/GDP
    debt: float = 0.6  # D/GDP
    primary_balance: float = 0.0  # (T-G)/GDP excluding interest

    # Policy stance
    fiscal_impulse: float = 0.0  # Change in structural balance
    automatic_stabilizer_effect: float = 0.0


@dataclass
class MonetaryState:
    """State of monetary policy."""

    policy_rate: float = 0.03
    real_rate: float = 0.01
    money_supply_growth: float = 0.05
    inflation: float = 0.02
    inflation_expectation: float = 0.02

    # Policy stance
    stance: str = "neutral"  # "tight", "neutral", "loose"


class FiscalPolicy:
    """
    Fiscal policy implementation following MMT principles.

    Key features:
    - Countercyclical spending rule
    - Automatic stabilizers
    - Employer of Last Resort (ELR)
    - Focus on real constraints (not debt ratios)
    """

    def __init__(
        self,
        g_bar: float = 0.2,
        alpha_gap: float = 0.3,
        delta_tension: float = 0.1,
        gamma_crisis: float = 0.4,
        tax_progressivity: float = 0.1,
        elr_enabled: bool = True,
        elr_wage_ratio: float = 0.8,
        seed: Optional[int] = None,
    ):
        self.g_bar = g_bar
        self.alpha_gap = alpha_gap
        self.delta_tension = delta_tension
        self.gamma_crisis = gamma_crisis
        self.tax_progressivity = tax_progressivity
        self.elr_enabled = elr_enabled
        self.elr_wage_ratio = elr_wage_ratio
        self.rng = np.random.default_rng(seed)

        self.state = FiscalState()

    def compute_spending(
        self,
        output_gap: float,
        t_adjusted: float,
        is_crisis: bool,
        capacity_utilization: float = 0.85,
    ) -> float:
        """
        Compute government spending using enhanced stabilizer.

        G = G_bar - α*(Y-Y_pot) + δ*T_adj + γ*I_crisis

        MMT perspective: Spending limited by real capacity, not financial.
        """
        # Base spending
        g = self.g_bar

        # Countercyclical response (negative gap → more spending)
        g -= self.alpha_gap * output_gap

        # Tension response (higher tension → more stabilization)
        g += self.delta_tension * t_adjusted

        # Crisis intervention
        if is_crisis:
            g += self.gamma_crisis

        # Capacity constraint (can't spend if no real resources)
        # Reduce spending if economy at full capacity
        if capacity_utilization > 0.95:
            capacity_penalty = (capacity_utilization - 0.95) / 0.05
            g *= 1 - 0.2 * capacity_penalty

        self.state.spending = np.clip(g, 0.1, 0.45)
        return self.state.spending

    def compute_taxes(
        self,
        gdp_ratio: float,  # Y / Y_potential
        base_rate: float = 0.25,
    ) -> float:
        """
        Compute tax revenue with automatic stabilizer effect.

        T = τ * (Y/Y_pot)^(1+η)

        Progressive taxation provides automatic stabilization:
        - Boom: Higher incomes → higher marginal rates → T rises faster than Y
        - Recession: Lower incomes → lower marginal rates → T falls faster than Y
        """
        effective_rate = base_rate * (gdp_ratio ** (1 + self.tax_progressivity))
        self.state.tax_revenue = effective_rate
        return self.state.tax_revenue

    def compute_elr_employment(
        self,
        private_employment: float,
        employment_target: float = 0.96,
    ) -> float:
        """
        Compute Employer of Last Resort employment.

        L_ELR = max(0, L_target - L_private)

        ELR provides:
        - Floor on employment
        - Price anchor through ELR wage
        - Automatic countercyclical spending
        """
        if not self.elr_enabled:
            return 0.0

        elr = max(0, employment_target - private_employment)
        return elr

    def compute_budget(
        self,
        gdp: float,
        interest_rate: float,
    ) -> Tuple[float, float, float]:
        """
        Compute full budget using operational constraint.

        G + i*D_{-1} = T + ΔD + ΔH

        Returns (deficit, primary_balance, new_debt_gdp)
        """
        # Interest payments
        interest = interest_rate * self.state.debt

        # Total spending (G + interest)
        total_spending = self.state.spending + interest

        # Deficit
        self.state.deficit = total_spending - self.state.tax_revenue

        # Primary balance (excluding interest)
        self.state.primary_balance = self.state.tax_revenue - self.state.spending

        # Debt dynamics (simplified)
        # dD/Y = deficit + (i - g)*D/Y
        # where g is nominal GDP growth
        gdp_growth = 0.03  # Placeholder
        debt_dynamics = self.state.deficit + (interest_rate - gdp_growth) * self.state.debt

        self.state.debt = max(0, self.state.debt + debt_dynamics)

        return self.state.deficit, self.state.primary_balance, self.state.debt

    def evaluate_sustainability(
        self,
        nominal_growth: float,
        interest_rate: float,
    ) -> Dict[str, float]:
        """
        Evaluate fiscal sustainability from MMT perspective.

        Key insight: For sovereign currency issuers, sustainability
        is about real resource constraints, not debt ratios.
        """
        # r - g differential
        r_g = interest_rate - nominal_growth

        # Stable debt ratio if primary_balance >= r_g * debt
        required_primary = r_g * self.state.debt
        sustainability_gap = self.state.primary_balance - required_primary

        # But from MMT view, focus on inflation and capacity
        return {
            "debt_gdp": self.state.debt,
            "r_minus_g": r_g,
            "sustainability_gap": sustainability_gap,
            "deficit_gdp": self.state.deficit,
            "is_sustainable_conventional": sustainability_gap >= 0,
            # MMT would say: sustainable if not causing inflation
        }


class MonetaryPolicy:
    """
    Monetary policy implementation.

    Taylor-like rule with modifications for MMT consistency:
    - Policy rate responds to inflation and output gap
    - But recognizes fiscal policy as primary stabilizer
    """

    def __init__(
        self,
        neutral_rate: float = 0.02,
        inflation_target: float = 0.02,
        inflation_weight: float = 1.5,
        output_weight: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.neutral_rate = neutral_rate
        self.inflation_target = inflation_target
        self.inflation_weight = inflation_weight
        self.output_weight = output_weight
        self.rng = np.random.default_rng(seed)

        self.state = MonetaryState()

    def compute_policy_rate(
        self,
        inflation: float,
        output_gap: float,
        financial_tension: float = 0.0,
    ) -> float:
        """
        Compute policy rate using modified Taylor rule.

        i = r* + π + φ_π*(π - π*) + φ_y*(y - y*)

        Modified for financial stability considerations.
        """
        inflation_gap = inflation - self.inflation_target

        # Taylor rule
        rate = (
            self.neutral_rate
            + inflation
            + self.inflation_weight * inflation_gap
            + self.output_weight * output_gap
        )

        # Financial stability adjustment
        # High financial tension → lower rates for stability
        # But not too low if already accommodative
        if financial_tension > 0.5:
            rate -= 0.01 * (financial_tension - 0.5)

        # Zero lower bound
        rate = max(0.0, rate)

        self.state.policy_rate = rate
        self.state.real_rate = rate - inflation

        # Determine stance
        if rate < self.neutral_rate - 0.01:
            self.state.stance = "loose"
        elif rate > self.neutral_rate + 0.01:
            self.state.stance = "tight"
        else:
            self.state.stance = "neutral"

        return rate

    def compute_inflation(
        self,
        output_gap: float,
        t_energy: float,
        t_trade: float,
        expected_inflation: float,
        coefficients: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute inflation as capacity phenomenon.

        π = β_0 + β_1*(Y/Y_pot) + β_2*T_E + β_3*T_C + β_4*E[π_{t+1}]

        From MMT: Inflation reflects real constraints and structural tensions,
        not just monetary expansion.
        """
        if coefficients is None:
            coefficients = np.array([0.02, 0.3, 0.1, 0.15, 0.25])

        inflation = (
            coefficients[0]  # Base inflation
            + coefficients[1] * max(0, output_gap)  # Demand-pull (only when positive gap)
            + coefficients[2] * t_energy  # Energy cost-push
            + coefficients[3] * t_trade  # Trade/supply chain
            + coefficients[4] * expected_inflation  # Expectations
        )

        self.state.inflation = max(-0.05, inflation)  # Floor at -5% deflation
        return self.state.inflation

    def update_expectations(
        self,
        realized_inflation: float,
        adaptive_weight: float = 0.3,
    ) -> float:
        """
        Update inflation expectations adaptively.

        E[π_{t+1}] = (1-λ)*E[π_t] + λ*π_t
        """
        self.state.inflation_expectation = (
            (1 - adaptive_weight) * self.state.inflation_expectation
            + adaptive_weight * realized_inflation
        )
        return self.state.inflation_expectation


class MMTStabilizers:
    """
    Combined fiscal-monetary stabilization system.

    Implements the full MMT-compatible policy framework:
    1. Fiscal policy as primary stabilizer
    2. ELR for full employment
    3. Monetary policy for inflation management
    4. Coordination of fiscal-monetary mix
    """

    def __init__(
        self,
        fiscal_params: Optional[Dict] = None,
        monetary_params: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        fiscal_params = fiscal_params or {}
        monetary_params = monetary_params or {}

        self.fiscal = FiscalPolicy(seed=seed, **fiscal_params)
        self.monetary = MonetaryPolicy(seed=seed, **monetary_params)

        # Coordination parameters
        self.fiscal_monetary_coordination = 0.3  # How much fiscal considers monetary

    def stabilize(
        self,
        output_gap: float,
        t_adjusted: float,
        t_energy: float,
        t_trade: float,
        capacity_utilization: float,
        private_employment: float,
        is_crisis: bool,
    ) -> Dict[str, float]:
        """
        Apply full stabilization framework.

        Returns all policy outputs and macro outcomes.
        """
        # Fiscal policy first (MMT emphasis)
        spending = self.fiscal.compute_spending(
            output_gap, t_adjusted, is_crisis, capacity_utilization
        )

        # ELR
        elr_employment = self.fiscal.compute_elr_employment(private_employment)

        # Taxes (automatic stabilizers)
        gdp_ratio = 1 + output_gap  # Approximate Y/Y_pot
        taxes = self.fiscal.compute_taxes(gdp_ratio)

        # Inflation (before monetary policy)
        expected_inflation = self.monetary.state.inflation_expectation
        inflation = self.monetary.compute_inflation(
            output_gap, t_energy, t_trade, expected_inflation
        )

        # Monetary policy (responds to fiscal stance and inflation)
        financial_tension = t_adjusted * 0.5  # Simplified
        policy_rate = self.monetary.compute_policy_rate(
            inflation, output_gap, financial_tension
        )

        # Budget computation
        deficit, primary_balance, debt = self.fiscal.compute_budget(
            gdp=1 + output_gap,  # Normalized
            interest_rate=policy_rate,
        )

        # Update inflation expectations
        self.monetary.update_expectations(inflation)

        return {
            "spending": spending,
            "taxes": taxes,
            "deficit": deficit,
            "debt": debt,
            "primary_balance": primary_balance,
            "elr_employment": elr_employment,
            "policy_rate": policy_rate,
            "inflation": inflation,
            "expected_inflation": expected_inflation,
            "fiscal_stance": "expansionary" if deficit > 0.03 else "contractionary" if deficit < -0.01 else "neutral",
            "monetary_stance": self.monetary.state.stance,
        }

    def evaluate_policy_space(
        self,
        capacity_utilization: float,
        inflation: float,
    ) -> Dict[str, str]:
        """
        Evaluate available policy space from MMT perspective.

        Real constraints (capacity, inflation) determine space, not debt.
        """
        # Fiscal space determined by real resources
        if capacity_utilization < 0.9 and inflation < 0.04:
            fiscal_space = "ample"
        elif capacity_utilization < 0.95 and inflation < 0.05:
            fiscal_space = "moderate"
        else:
            fiscal_space = "limited"

        # Monetary space determined by rate level
        if self.monetary.state.policy_rate > 0.02:
            monetary_space = "ample"
        elif self.monetary.state.policy_rate > 0:
            monetary_space = "moderate"
        else:
            monetary_space = "limited_zlb"  # Zero lower bound

        # Combined assessment
        if fiscal_space == "ample":
            combined = "expand_fiscal"
        elif fiscal_space == "limited" and inflation > 0.05:
            combined = "tighten_both"
        elif monetary_space == "limited_zlb":
            combined = "fiscal_primary"
        else:
            combined = "coordinate"

        return {
            "fiscal_space": fiscal_space,
            "monetary_space": monetary_space,
            "recommended_policy": combined,
        }

    def to_dict(self) -> Dict[str, float]:
        """Export current policy state."""
        return {
            **{f"fiscal_{k}": v for k, v in vars(self.fiscal.state).items()},
            **{f"monetary_{k}": v for k, v in vars(self.monetary.state).items()},
        }
