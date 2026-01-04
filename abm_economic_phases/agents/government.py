"""
Government Agent

Implements government behavior from Table 1:
- Function: Fiscal/Monetary policy
- Behavior rule: G_t = G_bar - alpha*(Y_t - Y_pot) + beta*T_adjusted

The government acts as a system stabilizer, implementing:
- Countercyclical fiscal policy
- Automatic stabilizers
- Employer of Last Resort (ELR) mechanism
- Coordination with monetary policy
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from .base import Agent, AgentType, AgentState, MacroState


@dataclass
class GovernmentState(AgentState):
    """Extended state for government agent."""

    # Fiscal position
    spending: float = 0.2  # G_t as fraction of GDP
    tax_revenue: float = 0.18  # T_t as fraction of GDP
    deficit: float = 0.02  # (G - T) as fraction of GDP
    debt_gdp: float = 0.6  # Debt/GDP ratio

    # Policy rates and targets
    policy_rate: float = 0.03  # Central bank rate
    inflation_target: float = 0.02
    unemployment_target: float = 0.04

    # Employer of Last Resort
    elr_employment: float = 0.0  # Public employment
    elr_wage: float = 0.8  # ELR wage (fraction of average)

    # Policy stance indicators
    fiscal_stance: float = 0.0  # Positive = expansionary
    monetary_stance: float = 0.0  # Positive = accommodative


class Government(Agent):
    """
    Government agent implementing fiscal and monetary policy.

    Key policy mechanisms:
    1. Countercyclical spending: G adjusts inversely to output gap
    2. Automatic stabilizers: Tax revenue fluctuates with income
    3. Employer of Last Resort: Public employment absorbs unemployment
    4. Monetary policy: Taylor-like rule for interest rates

    Compatible with Modern Monetary Theory (MMT) perspective:
    - Fiscal space constrained by real resources, not finance
    - Inflation as the true constraint
    - Government as system stabilizer
    """

    def __init__(
        self,
        agent_id: int = 0,
        g_bar: float = 0.2,
        alpha_gap: float = 0.3,
        delta_tension: float = 0.1,
        gamma_crisis: float = 0.4,
        inflation_weight: float = 1.5,
        output_weight: float = 0.5,
        elr_enabled: bool = True,
        noise_std: float = 0.005,
        seed: Optional[int] = None,
    ):
        # Fiscal policy parameters
        self.g_bar = g_bar  # Base spending/GDP
        self.alpha_gap = alpha_gap  # Response to output gap
        self.delta_tension = delta_tension  # Response to tensions
        self.gamma_crisis = gamma_crisis  # Crisis intervention

        # Monetary policy parameters (Taylor rule)
        self.inflation_weight = inflation_weight  # phi_pi
        self.output_weight = output_weight  # phi_y

        # ELR mechanism
        self.elr_enabled = elr_enabled

        super().__init__(agent_id, AgentType.GOVERNMENT, noise_std, seed)

    def _initialize_state(self) -> None:
        """Initialize government state."""
        self.state = GovernmentState(
            wealth=0.0,  # Government "wealth" is not meaningful
            is_active=True,
        )

    def _compute_output_gap(self, macro_state: MacroState) -> float:
        """Compute output gap: (Y - Y_pot) / Y_pot."""
        return (macro_state.gdp - macro_state.potential_output) / max(
            macro_state.potential_output, 0.01
        )

    def _compute_fiscal_stance(self, macro_state: MacroState) -> float:
        """
        Determine fiscal policy stance based on macro conditions.

        Returns a value where:
        - Positive = expansionary (increase spending)
        - Negative = contractionary (reduce spending)
        """
        output_gap = self._compute_output_gap(macro_state)
        tension = np.mean(macro_state.tensions)

        # Base countercyclical response
        stance = -self.alpha_gap * output_gap

        # Add response to tensions
        stance += self.delta_tension * tension

        # Crisis intervention
        if macro_state.phase == "crisis":
            stance += self.gamma_crisis

        # Inflation constraint (MMT perspective)
        inflation_gap = macro_state.inflation - self.state.inflation_target
        if inflation_gap > 0.02:  # Significant inflation
            stance -= 0.2 * inflation_gap  # Reduce expansionary stance

        return np.clip(stance, -0.15, 0.25)

    def _compute_monetary_stance(self, macro_state: MacroState) -> float:
        """
        Determine monetary policy stance (Taylor rule variant).

        i = r* + pi + phi_pi*(pi - pi*) + phi_y*(y - y*)
        """
        output_gap = self._compute_output_gap(macro_state)
        inflation_gap = macro_state.inflation - self.state.inflation_target

        # Natural rate (time-varying)
        r_star = 0.02 + 0.5 * macro_state.gdp_growth

        # Taylor rule
        policy_rate = (
            r_star
            + macro_state.inflation
            + self.inflation_weight * inflation_gap
            + self.output_weight * output_gap
        )

        # Zero lower bound
        policy_rate = max(0.0, policy_rate)

        # Phase adjustments
        if macro_state.phase == "crisis":
            policy_rate = max(0.0, policy_rate - 0.02)  # Extra accommodation
        elif macro_state.phase == "overheating":
            policy_rate += 0.01  # Extra tightening

        return policy_rate

    def decide(
        self,
        macro_state: MacroState,
        neighbor_info: Dict[int, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Make fiscal and monetary policy decisions.

        Implements equations (18), (19), and (20) from the paper:
        - Operational budget constraint: G + i*D_{-1} = T + dD + dH
        - Inflation as capacity phenomenon
        - Enhanced automatic stabilizer
        """
        state = self.state

        # Compute policy stances
        fiscal_stance = self._compute_fiscal_stance(macro_state)
        policy_rate = self._compute_monetary_stance(macro_state)

        # Government spending
        spending = self.g_bar + fiscal_stance

        # Crisis indicator
        is_crisis = 1.0 if macro_state.phase == "crisis" else 0.0

        # Full spending rule: G = G_bar - alpha*(Y-Y_pot) + delta*T_adj - gamma*I_crisis
        tension_adjustment = self.delta_tension * np.mean(macro_state.tensions)
        crisis_adjustment = self.gamma_crisis * is_crisis

        spending = (
            self.g_bar
            - self.alpha_gap * self._compute_output_gap(macro_state)
            + tension_adjustment
            + crisis_adjustment
        )

        spending = np.clip(spending, 0.1, 0.4)  # Bounds on spending/GDP

        # Tax revenue (automatic stabilizer)
        # Progressive taxation: T = tau * Y^(1+eta)
        tax_rate_base = 0.25
        progressivity = 0.1
        gdp_ratio = macro_state.gdp / macro_state.potential_output

        tax_revenue = tax_rate_base * (gdp_ratio ** (1 + progressivity))

        # Deficit
        deficit = spending - tax_revenue

        # ELR employment
        if self.elr_enabled:
            private_employment = 1 - macro_state.unemployment
            employment_target = 1 - state.unemployment_target

            elr_employment = max(0, employment_target - private_employment)
            elr_cost = elr_employment * state.elr_wage * 0.1  # As fraction of GDP

            # Add ELR to spending
            spending += elr_cost
            deficit += elr_cost
        else:
            elr_employment = 0.0

        actions = {
            "spending": spending,
            "tax_revenue": tax_revenue,
            "deficit": deficit,
            "policy_rate": policy_rate,
            "elr_employment": elr_employment,
            "fiscal_stance": fiscal_stance,
        }

        self.action_history.append(actions)
        return actions

    def update(
        self,
        actions: Dict[str, float],
        macro_state: MacroState,
        market_outcomes: Dict[str, float],
    ) -> None:
        """Update government state."""
        state = self.state

        # Update fiscal variables
        state.spending = actions["spending"]
        state.tax_revenue = actions["tax_revenue"]
        state.deficit = actions["deficit"]

        # Update policy rate
        state.policy_rate = actions["policy_rate"]

        # Update ELR
        state.elr_employment = actions["elr_employment"]

        # Update debt/GDP
        # dD/GDP = deficit + i*D/GDP - g*D/GDP
        nominal_gdp_growth = macro_state.gdp_growth + macro_state.inflation
        interest_burden = state.policy_rate * state.debt_gdp
        debt_dynamics = (
            state.deficit + interest_burden - nominal_gdp_growth * state.debt_gdp
        )

        state.debt_gdp = max(0, state.debt_gdp + debt_dynamics)

        # Update stance indicators
        state.fiscal_stance = actions["fiscal_stance"]
        state.monetary_stance = actions["policy_rate"] - 0.03  # Deviation from neutral

    def compute_reward(
        self,
        actions: Dict[str, float],
        outcomes: Dict[str, float],
    ) -> float:
        """
        Compute policy success based on macro outcomes.

        Reward for:
        - Low unemployment
        - Inflation near target
        - Stable growth
        - Sustainable debt
        """
        unemployment_cost = -(outcomes.get("unemployment", 0.05) ** 2) * 10
        inflation_gap = abs(
            outcomes.get("inflation", 0.02) - self.state.inflation_target
        )
        inflation_cost = -(inflation_gap ** 2) * 5
        growth_reward = outcomes.get("gdp_growth", 0.02) * 2
        debt_cost = -max(0, self.state.debt_gdp - 1.0) * 0.1

        return unemployment_cost + inflation_cost + growth_reward + debt_cost

    def get_info(self) -> Dict[str, Any]:
        """Get government policy information."""
        return {
            "type": "government",
            "spending": self.state.spending,
            "deficit": self.state.deficit,
            "policy_rate": self.state.policy_rate,
            "debt_gdp": self.state.debt_gdp,
            "fiscal_stance": self.state.fiscal_stance,
            "elr_employment": self.state.elr_employment,
        }
