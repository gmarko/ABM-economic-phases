"""
Firm Agent

Implements firm behavior from Table 1:
- Function: Production, Investment, Employment
- Behavior rule: max pi_t = p_t * y_t - w_t * l_t - i_t * k_t

Firms make decisions about:
- Production level
- Labor hiring
- Capital investment
- Pricing (markup over marginal cost)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from .base import Agent, AgentType, AgentState, MacroState


@dataclass
class FirmState(AgentState):
    """Extended state for firm agents."""

    # Production
    output: float = 1.0  # y_t
    capital: float = 1.0  # k_t
    labor_demand: float = 1.0  # l_t

    # Pricing
    price: float = 1.0  # p_t
    markup: float = 0.15  # mu_t

    # Financial
    profits: float = 0.0  # pi_t
    investment: float = 0.1  # i_t
    credit_demand: float = 0.0
    credit_received: float = 0.0

    # Productivity
    tfp: float = 1.0  # Total factor productivity A_t

    # Sector classification
    sector: str = "manufacturing"


class Firm(Agent):
    """
    Firm agent implementing production and investment decisions.

    Production function: Y = A * K^alpha * L^(1-alpha) (Cobb-Douglas)
    Profit: pi = p*Y - w*L - r*K - investment costs
    Investment decision: Tobin's Q approximation
    """

    def __init__(
        self,
        agent_id: int,
        alpha: float = 0.33,  # Capital share
        depreciation: float = 0.025,
        markup_mean: float = 0.15,
        markup_std: float = 0.05,
        adjustment_cost: float = 0.5,
        noise_std: float = 0.03,
        sector: str = "manufacturing",
        seed: Optional[int] = None,
    ):
        self.alpha = alpha
        self.depreciation = depreciation
        self.markup_mean = markup_mean
        self.markup_std = markup_std
        self.adjustment_cost = adjustment_cost
        self.sector = sector

        super().__init__(agent_id, AgentType.FIRM, noise_std, seed)

    def _initialize_state(self) -> None:
        """Initialize firm-specific state."""
        # Random initial productivity and capital
        tfp = self.rng.lognormal(0, 0.2)
        capital = self.rng.lognormal(0, 0.3)

        self.state = FirmState(
            wealth=capital * 0.5,  # Equity
            capital=capital,
            tfp=tfp,
            markup=max(0.05, self.rng.normal(self.markup_mean, self.markup_std)),
            sector=self.sector,
            output=0.0,
            labor_demand=0.0,
        )

    def _production_function(self, capital: float, labor: float, tfp: float) -> float:
        """Cobb-Douglas production function."""
        return tfp * (capital ** self.alpha) * (labor ** (1 - self.alpha))

    def _marginal_product_labor(
        self, capital: float, labor: float, tfp: float
    ) -> float:
        """Marginal product of labor."""
        if labor <= 0:
            return float("inf")
        return (1 - self.alpha) * tfp * (capital / labor) ** self.alpha

    def _marginal_product_capital(
        self, capital: float, labor: float, tfp: float
    ) -> float:
        """Marginal product of capital."""
        if capital <= 0:
            return float("inf")
        return self.alpha * tfp * (labor / capital) ** (1 - self.alpha)

    def decide(
        self,
        macro_state: MacroState,
        neighbor_info: Dict[int, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Make production, hiring, and investment decisions.

        Based on profit maximization with adjustment costs and expectations.
        """
        state = self.state

        # Aggregate neighbor information (other firms)
        neighbor_agg = self.aggregate_neighbor_info(neighbor_info)

        # Expected demand based on macro state and network
        base_demand = macro_state.gdp * (1 + macro_state.gdp_growth)
        demand_uncertainty = np.sum(macro_state.tensions[:4]) / 4.0

        # Adjust expectations based on phase
        phase_multiplier = {
            "activation": 0.95,
            "expansion": 1.1,
            "maturity": 1.0,
            "overheating": 1.15,
            "crisis": 0.7,
            "recession": 0.85,
        }.get(macro_state.phase, 1.0)

        expected_demand = base_demand * phase_multiplier * (1 + self.get_noise())

        # Current wage (from macro state or estimated)
        wage = 1.0 + macro_state.inflation  # Real wage normalized

        # Optimal labor demand: MPL = w/p => L = ((1-alpha)*A*(K^alpha)/w)^(1/alpha)
        price = state.price
        optimal_labor = (
            (1 - self.alpha) * state.tfp * (state.capital ** self.alpha) / (wage / price)
        ) ** (1 / self.alpha)

        # Adjust for expected demand
        capacity_output = self._production_function(
            state.capital, optimal_labor, state.tfp
        )
        utilization = min(1.0, expected_demand / max(capacity_output, 0.01))

        labor_demand = optimal_labor * utilization * (1 + 0.1 * self.get_noise())
        labor_demand = max(0.1, labor_demand)

        # Production decision
        output = self._production_function(state.capital, labor_demand, state.tfp)

        # Pricing decision (markup over marginal cost)
        marginal_cost = wage / self._marginal_product_labor(
            state.capital, labor_demand, state.tfp
        )

        # Markup adjustment based on demand conditions
        if macro_state.phase == "overheating":
            markup_adj = 1.1  # More pricing power
        elif macro_state.phase == "crisis":
            markup_adj = 0.85  # Price competition
        else:
            markup_adj = 1.0

        target_markup = state.markup * markup_adj
        new_price = marginal_cost * (1 + target_markup)

        # Investment decision (simplified Tobin's Q)
        mpk = self._marginal_product_capital(state.capital, labor_demand, state.tfp)
        user_cost = macro_state.interest_rate + self.depreciation

        tobins_q = mpk / user_cost

        # Investment increases with Q > 1, decreases with Q < 1
        base_investment = self.depreciation * state.capital  # Replacement
        net_investment = (tobins_q - 1) * state.capital / (2 * self.adjustment_cost)

        # Phase-dependent investment behavior
        if macro_state.phase == "crisis":
            net_investment *= 0.3  # Sharp reduction
        elif macro_state.phase == "expansion":
            net_investment *= 1.2  # Accelerated investment

        # Uncertainty reduces investment
        net_investment *= 1 - 0.5 * demand_uncertainty

        total_investment = max(0, base_investment + net_investment)

        # Credit demand if investment exceeds internal funds
        internal_funds = max(0, state.profits + state.wealth * 0.1)
        credit_demand = max(0, total_investment - internal_funds)

        actions = {
            "output": output,
            "labor_demand": labor_demand,
            "price": new_price,
            "investment": total_investment,
            "credit_demand": credit_demand,
        }

        self.action_history.append(actions)
        return actions

    def update(
        self,
        actions: Dict[str, float],
        macro_state: MacroState,
        market_outcomes: Dict[str, float],
    ) -> None:
        """Update firm state after market clearing."""
        state = self.state

        prev_wealth = state.wealth

        # Update production variables
        state.output = actions["output"]
        state.labor_demand = actions["labor_demand"]
        state.price = actions["price"]
        state.investment = actions["investment"]
        state.credit_demand = actions["credit_demand"]

        # Credit received (may be rationed)
        credit_ratio = market_outcomes.get("credit_ratio", 1.0)
        state.credit_received = state.credit_demand * credit_ratio

        # Actual investment (may be lower if credit rationed)
        actual_investment = min(
            state.investment,
            state.credit_received + max(0, state.profits) + state.wealth * 0.1,
        )

        # Revenue and costs
        sales = state.output * state.price * market_outcomes.get("demand_ratio", 1.0)
        wage_bill = state.labor_demand * (1 + macro_state.inflation)
        interest_cost = macro_state.interest_rate * state.debt

        # Profits
        state.profits = sales - wage_bill - interest_cost - actual_investment * 0.5

        # Update capital stock
        state.capital = (1 - self.depreciation) * state.capital + actual_investment

        # Update debt
        state.debt = state.debt + state.credit_received - 0.1 * max(0, state.profits)
        state.debt = max(0, state.debt)

        # Update wealth (equity)
        state.wealth = state.capital - state.debt
        state.wealth = max(0.01, state.wealth)  # Prevent negative equity

        # Bankruptcy check
        if state.wealth < 0.01 or state.profits < -0.5 * state.capital:
            state.is_bankrupt = True
            state.is_active = False

        # TFP evolution (slow random walk with sector shocks)
        tfp_shock = self.get_noise() * 0.01
        if macro_state.phase == "expansion":
            tfp_shock += 0.002  # Learning by doing
        state.tfp *= 1 + tfp_shock
        state.tfp = max(0.5, min(2.0, state.tfp))

        # Update memory
        wealth_change = state.wealth - prev_wealth
        self.update_memory(wealth_change)

    def compute_reward(
        self,
        actions: Dict[str, float],
        outcomes: Dict[str, float],
    ) -> float:
        """Compute profit-based reward for the firm."""
        return self.state.profits + outcomes.get("wealth_change", 0.0)

    def get_info(self) -> Dict[str, Any]:
        """Get information to share with neighbors."""
        base_info = super().get_info()
        base_info.update({
            "output": self.state.output,
            "price": self.state.price,
            "sector": self.state.sector,
            "is_bankrupt": self.state.is_bankrupt,
        })
        return base_info
