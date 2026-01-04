"""
Household Agent

Implements household behavior from Table 1:
- Function: Consumption, Saving, Labor supply
- Behavior rule: max E[u(c_t, l_t)] s.t. c_t + s_t <= w_t * l_t + r_t * a_{t-1}

Households make decisions about:
- Consumption vs saving
- Labor supply
- Portfolio allocation (if applicable)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from .base import Agent, AgentType, AgentState, MacroState


@dataclass
class HouseholdState(AgentState):
    """Extended state for household agents."""

    # Labor
    labor_supply: float = 1.0  # l_t in [0, max_labor]
    max_labor: float = 1.0
    wage: float = 1.0  # w_t

    # Consumption and saving
    consumption: float = 0.0  # c_t
    savings: float = 0.0  # s_t (flow)
    assets: float = 1.0  # a_t (stock)

    # Expectations
    expected_income: float = 1.0
    expected_inflation: float = 0.02


class Household(Agent):
    """
    Household agent implementing consumption-saving decisions.

    Utility function: u(c, l) = c^(1-sigma)/(1-sigma) - chi * l^(1+eta)/(1+eta)
    Budget constraint: c + s = w*l + r*a_{-1}
    Asset accumulation: a = (1+r)*a_{-1} + s
    """

    def __init__(
        self,
        agent_id: int,
        risk_aversion: float = 2.0,
        discount_factor: float = 0.96,
        labor_elasticity: float = 0.5,
        labor_disutility: float = 1.0,
        noise_std: float = 0.02,
        seed: Optional[int] = None,
    ):
        # Preference parameters
        self.risk_aversion = risk_aversion  # sigma
        self.discount_factor = discount_factor  # beta
        self.labor_elasticity = labor_elasticity  # eta
        self.labor_disutility = labor_disutility  # chi

        super().__init__(agent_id, AgentType.HOUSEHOLD, noise_std, seed)

    def _initialize_state(self) -> None:
        """Initialize household-specific state."""
        self.state = HouseholdState(
            wealth=self.rng.uniform(0.5, 2.0),
            income=self.rng.uniform(0.8, 1.2),
            assets=self.rng.uniform(0.5, 2.0),
            labor_supply=self.rng.uniform(0.8, 1.0),
            wage=1.0,
            is_employed=True,
        )

    def decide(
        self,
        macro_state: MacroState,
        neighbor_info: Dict[int, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Make consumption, saving, and labor supply decisions.

        Implements Euler equation approximation with adaptive expectations.
        """
        state = self.state

        # Get aggregated neighbor information
        neighbor_agg = self.aggregate_neighbor_info(neighbor_info)

        # Adjust expectations based on macro state and neighbors
        expected_growth = 1 + macro_state.gdp_growth + self.get_noise()
        uncertainty = (
            np.sum(macro_state.tensions) / 5.0 + (1 - neighbor_agg.get("active_ratio", 1.0))
        )

        # Precautionary saving motive increases with uncertainty
        precautionary_factor = 1 + 0.2 * uncertainty

        # Available resources
        if state.is_employed:
            labor_income = state.wage * state.labor_supply
        else:
            # Unemployment benefit (fraction of previous wage)
            labor_income = 0.4 * state.wage

        capital_income = macro_state.interest_rate * state.assets
        total_resources = labor_income + capital_income + state.assets

        # Consumption decision (Euler equation approximation)
        # MRS = beta * (1+r) * E[c_{t+1}^{-sigma}] / c_t^{-sigma}
        # At optimum, MRS = 1
        intertemporal_factor = (
            self.discount_factor * (1 + macro_state.interest_rate) * expected_growth
        ) ** (1 / self.risk_aversion)

        # Target consumption rate
        target_consumption_rate = 1.0 / (1.0 + precautionary_factor / intertemporal_factor)

        # Phase-dependent adjustments
        if macro_state.phase == "crisis":
            target_consumption_rate *= 0.85  # Fear-driven reduction
        elif macro_state.phase == "expansion":
            target_consumption_rate *= 1.05  # Confidence-driven increase
        elif macro_state.phase == "overheating":
            target_consumption_rate *= 1.1  # FOMO effect

        # Actual consumption (bounded)
        consumption = np.clip(
            target_consumption_rate * total_resources,
            0.1 * state.income,  # Minimum subsistence
            0.95 * total_resources,  # Can't consume more than resources
        )

        # Saving is residual
        savings = total_resources - consumption - state.assets

        # Labor supply decision (if employed)
        if state.is_employed:
            # FOC: chi * l^eta = w * c^{-sigma}
            marginal_utility_consumption = consumption ** (-self.risk_aversion)
            optimal_labor = (
                state.wage * marginal_utility_consumption / self.labor_disutility
            ) ** (1 / self.labor_elasticity)

            # Adjust for macro conditions
            if macro_state.phase == "crisis":
                # Job insecurity -> work harder
                optimal_labor *= 1.1
            elif macro_state.unemployment > 0.08:
                # Fear of unemployment
                optimal_labor *= 1.05

            labor_supply = np.clip(optimal_labor + self.get_noise(), 0.0, state.max_labor)
        else:
            # Unemployed: job search intensity
            labor_supply = np.clip(0.5 + 0.3 * uncertainty, 0.1, 1.0)

        actions = {
            "consumption": consumption,
            "savings": savings,
            "labor_supply": labor_supply,
        }

        self.action_history.append(actions)
        return actions

    def update(
        self,
        actions: Dict[str, float],
        macro_state: MacroState,
        market_outcomes: Dict[str, float],
    ) -> None:
        """Update household state after market clearing."""
        state = self.state

        # Update employment status
        if state.is_employed:
            # Probability of job loss depends on macro conditions
            job_loss_prob = 0.02 + 0.1 * macro_state.tensions[3]  # Financial tension
            if macro_state.phase == "crisis":
                job_loss_prob += 0.1
            state.is_employed = self.rng.random() > job_loss_prob
        else:
            # Probability of finding job
            job_find_prob = 0.15 - 0.5 * macro_state.unemployment
            job_find_prob = max(0.05, job_find_prob)
            state.is_employed = self.rng.random() < job_find_prob

        # Update wage (if employed)
        if state.is_employed:
            wage_growth = (
                macro_state.inflation
                + 0.5 * macro_state.gdp_growth
                - 0.3 * macro_state.unemployment
            )
            state.wage *= 1 + wage_growth + 0.02 * self.get_noise()
            state.wage = max(0.5, state.wage)

        # Update financial position
        prev_wealth = state.wealth
        state.consumption = actions["consumption"]
        state.savings = actions["savings"]
        state.labor_supply = actions["labor_supply"]

        # Income this period
        if state.is_employed:
            state.income = state.wage * state.labor_supply
        else:
            state.income = 0.4 * state.wage  # Unemployment benefit

        # Asset accumulation
        state.assets = (1 + macro_state.interest_rate) * state.assets + state.savings

        # Wealth update
        state.wealth = state.assets

        # Update memory based on wealth change
        wealth_change = state.wealth - prev_wealth
        self.update_memory(wealth_change)

    def compute_reward(
        self,
        actions: Dict[str, float],
        outcomes: Dict[str, float],
    ) -> float:
        """Compute utility-based reward for the household."""
        c = actions.get("consumption", 1.0)
        l = actions.get("labor_supply", 0.5)

        # CRRA utility from consumption
        if self.risk_aversion == 1:
            utility_c = np.log(max(c, 0.01))
        else:
            utility_c = (max(c, 0.01) ** (1 - self.risk_aversion)) / (
                1 - self.risk_aversion
            )

        # Disutility from labor
        disutility_l = self.labor_disutility * (l ** (1 + self.labor_elasticity)) / (
            1 + self.labor_elasticity
        )

        return utility_c - disutility_l + outcomes.get("wealth_change", 0.0)

    def get_info(self) -> Dict[str, Any]:
        """Get information to share with neighbors."""
        base_info = super().get_info()
        base_info.update({
            "consumption": self.state.consumption,
            "is_employed": self.state.is_employed,
            "assets": self.state.assets,
        })
        return base_info
