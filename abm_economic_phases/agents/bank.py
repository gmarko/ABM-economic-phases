"""
Bank Agent

Implements bank behavior from Table 1:
- Function: Credit, Intermediation
- Behavior rule: Risk management sigma_t = f(PD_t, LGD_t, M_t)

Banks make decisions about:
- Credit supply to firms
- Interest rate spread
- Risk assessment
- Capital buffer management
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

from .base import Agent, AgentType, AgentState, MacroState


@dataclass
class BankState(AgentState):
    """Extended state for bank agents."""

    # Balance sheet
    assets: float = 10.0  # Loans + reserves
    loans: float = 8.0  # Outstanding credit
    deposits: float = 9.0  # Liabilities to depositors
    capital: float = 1.0  # Equity buffer
    reserves: float = 2.0  # Liquid reserves

    # Risk metrics
    capital_ratio: float = 0.1  # capital / risk-weighted assets
    reserve_ratio: float = 0.2  # reserves / deposits
    npl_ratio: float = 0.02  # Non-performing loans ratio
    expected_pd: float = 0.02  # Expected probability of default
    expected_lgd: float = 0.4  # Expected loss given default

    # Pricing
    lending_rate: float = 0.05  # Interest charged on loans
    deposit_rate: float = 0.02  # Interest paid on deposits
    spread: float = 0.03  # Lending - deposit rate

    # Credit supply
    credit_supply: float = 0.0
    credit_demand_received: float = 0.0


class Bank(Agent):
    """
    Bank agent implementing credit intermediation.

    Key behaviors:
    - Assess borrower creditworthiness
    - Set lending rates with risk premium
    - Manage capital and liquidity buffers
    - Respond to macroprudential regulations
    """

    def __init__(
        self,
        agent_id: int,
        min_capital_ratio: float = 0.08,
        target_capital_ratio: float = 0.12,
        min_reserve_ratio: float = 0.1,
        risk_aversion: float = 2.0,
        noise_std: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.min_capital_ratio = min_capital_ratio
        self.target_capital_ratio = target_capital_ratio
        self.min_reserve_ratio = min_reserve_ratio
        self.risk_aversion = risk_aversion

        super().__init__(agent_id, AgentType.BANK, noise_std, seed)

    def _initialize_state(self) -> None:
        """Initialize bank-specific state."""
        capital = self.rng.uniform(0.8, 1.5)
        loans = capital / self.target_capital_ratio
        deposits = loans * 0.9
        reserves = deposits * 0.2

        self.state = BankState(
            wealth=capital,
            capital=capital,
            loans=loans,
            deposits=deposits,
            reserves=reserves,
            assets=loans + reserves,
            capital_ratio=capital / loans,
            reserve_ratio=reserves / deposits,
        )

    def _assess_credit_risk(
        self,
        macro_state: MacroState,
        borrower_info: Dict[str, Any],
    ) -> float:
        """
        Assess credit risk for a potential borrower.

        Returns expected loss = PD * LGD * EAD
        """
        # Base probability of default from macro conditions
        base_pd = 0.02

        # Adjust for financial tension
        pd_tension = base_pd + 0.05 * macro_state.tensions[3]

        # Phase-dependent adjustment
        phase_pd = {
            "activation": 0.03,
            "expansion": 0.02,
            "maturity": 0.025,
            "overheating": 0.04,
            "crisis": 0.12,
            "recession": 0.08,
        }.get(macro_state.phase, 0.03)

        # Borrower-specific factors (if available)
        borrower_pd_adj = 0.0
        if borrower_info:
            if borrower_info.get("is_bankrupt", False):
                return 1.0  # Maximum risk
            leverage = borrower_info.get("debt", 0) / max(
                borrower_info.get("wealth", 1), 0.01
            )
            borrower_pd_adj = 0.02 * max(0, leverage - 0.5)

        final_pd = min(0.5, pd_tension + phase_pd + borrower_pd_adj)

        # Expected LGD
        lgd = 0.4 + 0.2 * macro_state.tensions[3]

        return final_pd * lgd

    def decide(
        self,
        macro_state: MacroState,
        neighbor_info: Dict[int, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Make credit supply and pricing decisions.

        Based on risk assessment and capital constraints.
        """
        state = self.state

        # Update risk estimates
        state.expected_pd = 0.02 + 0.1 * macro_state.tensions[3]
        if macro_state.phase == "crisis":
            state.expected_pd += 0.05

        # Calculate available lending capacity
        # Constrained by capital ratio requirement
        max_rwa = state.capital / self.min_capital_ratio
        current_rwa = state.loans  # Simplified: RWA = loans

        # Lending headroom
        lending_capacity = max(0, max_rwa - current_rwa)

        # Adjust for macroprudential buffer
        if macro_state.phase in ["overheating", "crisis"]:
            # Countercyclical buffer
            lending_capacity *= 0.8

        # Expected credit demand (from network)
        borrower_firms = [
            info for info in neighbor_info.values()
            if info.get("type") == "firm" and info.get("is_active", True)
        ]

        expected_demand = sum(
            info.get("credit_demand", 0) for info in borrower_firms
        )

        # Credit supply decision
        # More conservative in high-tension periods
        risk_adjustment = 1 - 0.5 * (state.expected_pd / 0.1)
        risk_adjustment = max(0.3, risk_adjustment)

        credit_supply = min(lending_capacity, expected_demand) * risk_adjustment

        # Pricing decision
        # Spread = risk premium + operating costs + profit margin
        risk_premium = state.expected_pd * state.expected_lgd * 2  # Risk charge
        operating_cost = 0.01
        profit_margin = 0.005 + 0.01 * (1 - credit_supply / max(lending_capacity, 1))

        new_spread = risk_premium + operating_cost + profit_margin + self.get_noise()
        new_spread = np.clip(new_spread, 0.01, 0.15)

        # Lending rate = policy rate + spread
        lending_rate = macro_state.interest_rate + new_spread

        # Deposit rate (slightly below policy rate)
        deposit_rate = max(0, macro_state.interest_rate - 0.01)

        actions = {
            "credit_supply": credit_supply,
            "lending_rate": lending_rate,
            "deposit_rate": deposit_rate,
            "spread": new_spread,
        }

        self.action_history.append(actions)
        return actions

    def update(
        self,
        actions: Dict[str, float],
        macro_state: MacroState,
        market_outcomes: Dict[str, float],
    ) -> None:
        """Update bank state after credit market clearing."""
        state = self.state

        prev_wealth = state.wealth

        # Update rates
        state.lending_rate = actions["lending_rate"]
        state.deposit_rate = actions["deposit_rate"]
        state.spread = actions["spread"]

        # New lending
        credit_extended = actions["credit_supply"] * market_outcomes.get(
            "credit_demand_ratio", 1.0
        )

        # Loan defaults
        default_rate = state.expected_pd + self.get_noise() * 0.02
        default_rate = np.clip(default_rate, 0, 0.3)

        loan_losses = state.loans * default_rate * state.expected_lgd

        # Update NPL ratio
        state.npl_ratio = default_rate

        # Interest income and expense
        interest_income = state.loans * state.lending_rate
        interest_expense = state.deposits * state.deposit_rate

        # Net interest margin
        net_income = interest_income - interest_expense - loan_losses

        # Update balance sheet
        # Loans: previous + new lending - defaults - repayments
        repayment_rate = 0.1  # 10% of loans repaid each period
        state.loans = (
            state.loans * (1 - default_rate - repayment_rate) + credit_extended
        )
        state.loans = max(0, state.loans)

        # Capital accumulation (retained earnings)
        state.capital = state.capital + net_income * 0.8  # 80% retention
        state.capital = max(0.01, state.capital)

        # Deposits (exogenous growth linked to economy)
        deposit_growth = 0.02 + macro_state.gdp_growth + self.get_noise() * 0.01
        state.deposits *= 1 + deposit_growth
        state.deposits = max(state.loans * 0.5, state.deposits)

        # Reserves
        state.reserves = state.deposits * self.min_reserve_ratio
        state.assets = state.loans + state.reserves

        # Update ratios
        state.capital_ratio = state.capital / max(state.loans, 0.01)
        state.reserve_ratio = state.reserves / max(state.deposits, 0.01)

        # Wealth = capital (equity)
        state.wealth = state.capital

        # Bank failure check
        if state.capital_ratio < self.min_capital_ratio * 0.5:
            state.is_bankrupt = True
            state.is_active = False

        # Update memory
        wealth_change = state.wealth - prev_wealth
        self.update_memory(wealth_change)

    def process_credit_requests(
        self,
        requests: List[Dict[str, Any]],
        macro_state: MacroState,
    ) -> Dict[int, float]:
        """
        Process credit requests from firms.

        Returns dict mapping borrower_id to credit granted.
        """
        state = self.state

        # Sort by creditworthiness
        scored_requests = []
        for req in requests:
            risk = self._assess_credit_risk(macro_state, req.get("borrower_info", {}))
            scored_requests.append((req, risk))

        scored_requests.sort(key=lambda x: x[1])  # Lower risk first

        # Allocate credit
        remaining_supply = state.credit_supply
        allocations = {}

        for req, risk in scored_requests:
            borrower_id = req.get("borrower_id")
            demand = req.get("amount", 0)

            if remaining_supply <= 0:
                allocations[borrower_id] = 0
                continue

            # Risk-based allocation
            if risk > 0.3:  # Too risky
                allocations[borrower_id] = 0
            else:
                granted = min(demand, remaining_supply * (1 - risk))
                allocations[borrower_id] = granted
                remaining_supply -= granted

        return allocations

    def compute_reward(
        self,
        actions: Dict[str, float],
        outcomes: Dict[str, float],
    ) -> float:
        """Compute reward based on risk-adjusted returns."""
        interest_margin = actions["spread"] * self.state.loans
        losses = self.state.npl_ratio * self.state.loans * 0.4
        capital_cost = max(0, self.target_capital_ratio - self.state.capital_ratio) * 0.1

        return interest_margin - losses - capital_cost

    def get_info(self) -> Dict[str, Any]:
        """Get information to share with neighbors."""
        base_info = super().get_info()
        base_info.update({
            "lending_rate": self.state.lending_rate,
            "credit_supply": self.state.credit_supply,
            "capital_ratio": self.state.capital_ratio,
            "npl_ratio": self.state.npl_ratio,
        })
        return base_info
