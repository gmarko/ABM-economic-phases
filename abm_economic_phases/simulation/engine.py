"""
Main Simulation Engine

Implements Algorithm 2 from the paper:
1. Initialize agents with random attributes
2. Initialize scale-free network
3. Initialize memory M_i^0 = 0
4. Set initial state S_0 = (Activation, T_0, A_0, 0)
5. For each time step:
   - Step 1: Local network interaction
   - Step 2: Macroeconomic aggregation
   - Step 3: External events and tensions
   - Step 4: Phase transition
   - Step 5: Learning and network evolution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm

from ..agents import Household, Firm, Bank, Government, AgentType
from ..agents.base import MacroState
from ..network import ScaleFreeNetwork
from ..economy import (
    EconomicPhase,
    PhaseTransitionEngine,
    GDPVector,
    TensionSystem,
    ExtremeEventGenerator,
    EventType,
)
from ..economy.phases import PhaseConditions
from ..learning import AdaptiveMemorySystem, SARSALambda
from ..mmt import MMTStabilizers
from ..utils.parameters import ModelParameters


@dataclass
class SimulationResults:
    """Results from a simulation run."""

    # Time series
    gdp: List[float] = field(default_factory=list)
    gdp_growth: List[float] = field(default_factory=list)
    unemployment: List[float] = field(default_factory=list)
    inflation: List[float] = field(default_factory=list)

    # Phase history
    phases: List[str] = field(default_factory=list)
    phase_durations: Dict[str, int] = field(default_factory=dict)

    # Tensions
    tensions: List[Dict[str, float]] = field(default_factory=list)

    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Memory
    memory: List[Dict[str, float]] = field(default_factory=list)

    # Policy
    policy: List[Dict[str, float]] = field(default_factory=list)

    # Network metrics over time
    network_metrics: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gdp": self.gdp,
            "gdp_growth": self.gdp_growth,
            "unemployment": self.unemployment,
            "inflation": self.inflation,
            "phases": self.phases,
            "phase_durations": self.phase_durations,
            "tensions": self.tensions,
            "events": self.events,
            "memory": self.memory,
            "policy": self.policy,
            "network_metrics": self.network_metrics,
        }


class EconomicSimulation:
    """
    Main simulation engine for the ABM Economic Model.

    Coordinates all components:
    - Heterogeneous agents
    - Scale-free network
    - Phase transitions
    - Tension dynamics
    - Extreme events
    - Adaptive memory
    - MMT stabilizers
    """

    def __init__(
        self,
        params: Optional[ModelParameters] = None,
        seed: Optional[int] = None,
    ):
        self.params = params or ModelParameters()
        self.seed = seed or self.params.seed
        self.rng = np.random.default_rng(self.seed)

        # Initialize components
        self._initialize_agents()
        self._initialize_network()
        self._initialize_systems()

        # Current state
        self.current_phase = EconomicPhase.ACTIVATION
        self.current_step = 0
        self.macro_state = MacroState()

        # Results storage
        self.results = SimulationResults()

    def _initialize_agents(self) -> None:
        """Initialize all agent populations."""
        agent_params = self.params.agents

        # Households
        self.households: Dict[int, Household] = {}
        for i in range(agent_params.n_households):
            self.households[i] = Household(
                agent_id=i,
                risk_aversion=agent_params.household_risk_aversion,
                discount_factor=agent_params.household_discount_factor,
                labor_elasticity=agent_params.labor_elasticity,
                noise_std=agent_params.noise_std["household"],
                seed=self.seed + i,
            )

        # Firms
        self.firms: Dict[int, Firm] = {}
        base_id = agent_params.n_households
        sectors = ["primary", "manufacturing", "services", "financial", "public"]
        for i in range(agent_params.n_firms):
            sector = sectors[i % len(sectors)]
            self.firms[base_id + i] = Firm(
                agent_id=base_id + i,
                alpha=agent_params.production_alpha,
                depreciation=agent_params.capital_depreciation,
                markup_mean=agent_params.firm_markup_mean,
                markup_std=agent_params.firm_markup_std,
                sector=sector,
                noise_std=agent_params.noise_std["firm"],
                seed=self.seed + base_id + i,
            )

        # Banks
        self.banks: Dict[int, Bank] = {}
        base_id = agent_params.n_households + agent_params.n_firms
        for i in range(agent_params.n_banks):
            self.banks[base_id + i] = Bank(
                agent_id=base_id + i,
                min_capital_ratio=agent_params.bank_capital_ratio,
                noise_std=agent_params.noise_std["bank"],
                seed=self.seed + base_id + i,
            )

        # Government
        gov_id = base_id + agent_params.n_banks
        self.government = Government(
            agent_id=gov_id,
            g_bar=self.params.mmt.g_bar,
            alpha_gap=self.params.mmt.alpha_gap,
            delta_tension=self.params.mmt.delta_tension,
            gamma_crisis=self.params.mmt.gamma_crisis,
            seed=self.seed + gov_id,
        )

    def _initialize_network(self) -> None:
        """Initialize scale-free network topology."""
        self.network = ScaleFreeNetwork(
            n_households=self.params.agents.n_households,
            n_firms=self.params.agents.n_firms,
            n_banks=self.params.agents.n_banks,
            m=self.params.network.m,
            k_0=self.params.network.k0,
            seed=self.seed,
        )

    def _initialize_systems(self) -> None:
        """Initialize subsystems."""
        # Phase transition engine
        self.phase_engine = PhaseTransitionEngine(
            transition_noise_std=self.params.phases.transition_noise_std,
            seed=self.seed,
        )

        # GDP vector
        self.gdp_vector = GDPVector()

        # Tension system
        self.tension_system = TensionSystem(
            memory_decay=self.params.memory.delta_m,
            learning_rate=self.params.tension_learning_rate,
            seed=self.seed,
        )

        # Event generator
        self.event_generator = ExtremeEventGenerator(seed=self.seed)

        # Memory system
        self.memory_system = AdaptiveMemorySystem(
            delta_m=self.params.memory.delta_m,
            tau=self.params.memory.tau,
            lambda_sector=self.params.memory.lambda_sector,
            gamma_systemic=self.params.memory.gamma_systemic,
            sector_weights=self.params.memory.sector_weights,
            seed=self.seed,
        )

        # Register agents with memory system
        for agent_id, firm in self.firms.items():
            self.memory_system.register_agent(agent_id, firm.sector)
        for agent_id in self.households:
            self.memory_system.register_agent(agent_id, "services")  # Default
        for agent_id in self.banks:
            self.memory_system.register_agent(agent_id, "financial")

        # MMT stabilizers
        self.mmt = MMTStabilizers(seed=self.seed)

    def _step_local_interaction(self) -> Dict[int, Dict[str, float]]:
        """
        Step 1: Local network interaction.

        Each agent observes neighbors and makes decisions.
        """
        all_actions = {}

        # Get network information for each agent
        def get_neighbor_info(agent_id: int) -> Dict[int, Dict[str, Any]]:
            neighbors = self.network.get_neighbors(agent_id)
            return {n: self._get_agent(n).get_info() for n in neighbors if self._get_agent(n)}

        # Households decide
        for agent_id, household in self.households.items():
            neighbor_info = get_neighbor_info(agent_id)
            actions = household.decide(self.macro_state, neighbor_info)
            all_actions[agent_id] = actions

        # Firms decide
        for agent_id, firm in self.firms.items():
            neighbor_info = get_neighbor_info(agent_id)
            actions = firm.decide(self.macro_state, neighbor_info)
            all_actions[agent_id] = actions

        # Banks decide
        for agent_id, bank in self.banks.items():
            neighbor_info = get_neighbor_info(agent_id)
            actions = bank.decide(self.macro_state, neighbor_info)
            all_actions[agent_id] = actions

        # Government decides
        gov_info = {}  # Government observes macro state directly
        gov_actions = self.government.decide(self.macro_state, gov_info)
        all_actions[self.government.agent_id] = gov_actions

        return all_actions

    def _step_aggregation(
        self,
        all_actions: Dict[int, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Step 2: Macroeconomic aggregation.

        Compute Y_t, U_t, π_t, C_t, I_t from agent actions.
        """
        # Total output (from firms)
        total_output = sum(
            actions.get("output", 0)
            for agent_id, actions in all_actions.items()
            if agent_id in self.firms
        )

        # Sector outputs
        sector_outputs = {}
        for agent_id, firm in self.firms.items():
            sector = firm.sector
            output = all_actions[agent_id].get("output", 0)
            sector_outputs[sector] = sector_outputs.get(sector, 0) + output

        # Total consumption (from households)
        total_consumption = sum(
            actions.get("consumption", 0)
            for agent_id, actions in all_actions.items()
            if agent_id in self.households
        )

        # Total investment (from firms)
        total_investment = sum(
            actions.get("investment", 0)
            for agent_id, actions in all_actions.items()
            if agent_id in self.firms
        )

        # Labor market
        total_labor_supply = sum(
            actions.get("labor_supply", 0)
            for agent_id, actions in all_actions.items()
            if agent_id in self.households
        )
        total_labor_demand = sum(
            actions.get("labor_demand", 0)
            for agent_id, actions in all_actions.items()
            if agent_id in self.firms
        )

        # Unemployment rate
        if total_labor_supply > 0:
            employment_rate = min(1, total_labor_demand / total_labor_supply)
            unemployment = 1 - employment_rate
        else:
            unemployment = 0.05

        # Credit market
        total_credit_supply = sum(
            actions.get("credit_supply", 0)
            for agent_id, actions in all_actions.items()
            if agent_id in self.banks
        )
        total_credit_demand = sum(
            actions.get("credit_demand", 0)
            for agent_id, actions in all_actions.items()
            if agent_id in self.firms
        )

        # Price level (average firm prices weighted by output)
        prices = []
        weights = []
        for agent_id, firm in self.firms.items():
            prices.append(all_actions[agent_id].get("price", 1.0))
            weights.append(all_actions[agent_id].get("output", 1.0))
        if sum(weights) > 0:
            avg_price = np.average(prices, weights=weights)
        else:
            avg_price = 1.0

        # Update GDP vector
        g, a, theta = self.gdp_vector.update(total_output, sector_outputs)

        # Inflation (from price changes)
        if len(self.results.gdp) > 0:
            prev_price = self.results.inflation[-1] + 1 if self.results.inflation else 1
            inflation = (avg_price / prev_price) - 1
        else:
            inflation = 0.02

        return {
            "gdp": total_output,
            "gdp_growth": g,
            "gdp_acceleration": a,
            "sectoral_coherence": theta,
            "consumption": total_consumption,
            "investment": total_investment,
            "unemployment": unemployment,
            "inflation": inflation,
            "credit_supply": total_credit_supply,
            "credit_demand": total_credit_demand,
            "avg_price": avg_price,
            "sector_outputs": sector_outputs,
        }

    def _step_events_tensions(
        self,
        aggregates: Dict[str, float],
    ) -> Tuple[Dict[str, float], Optional[Any]]:
        """
        Step 3: External events and tensions.

        Generate events and update tensions.
        """
        # Update tensions
        macro_memory = self.memory_system.state.macro_memory

        # Use aggregates to inform tension updates
        financial_inputs = {
            "corporate_spread": 0.02 + 0.05 * self.macro_state.tensions[3],
            "aggregate_leverage": 0.5,
            "credit_growth": aggregates.get("gdp_growth", 0.02),
            "npl_ratio": 0.02,
        }

        tensions = self.tension_system.update(
            macro_memory=macro_memory,
            financial_inputs=financial_inputs,
        )

        # Generate potential extreme event
        event = self.event_generator.generate_event(
            t_adjusted=tensions.t_adjusted,
            macro_memory=macro_memory,
            coupling=self.macro_state.coupling,
            sectoral_coherence=aggregates.get("sectoral_coherence", 0.7),
            financial_tension=tensions.t_financial,
        )

        # Apply event impact if occurred
        if event:
            # Propagate through network
            network_metrics = self.network.compute_metrics()
            impact = self.event_generator.compute_systemic_impact(
                event,
                {
                    "average_degree": network_metrics.average_degree,
                    "max_degree": network_metrics.max_degree,
                    "clustering": network_metrics.avg_clustering,
                }
            )

            # Update tensions based on event
            self.tension_system.simulate_shock(
                event.tension_target,
                abs(event.effective_impact),
            )

        return self.tension_system.to_dict(), event

    def _step_phase_transition(
        self,
        aggregates: Dict[str, float],
        tensions: Dict[str, float],
        event: Optional[Any],
    ) -> EconomicPhase:
        """
        Step 4: Phase transition evaluation.

        Determine if phase should change based on conditions.
        """
        # Build phase conditions
        conditions = PhaseConditions(
            g_t=aggregates.get("gdp_growth", 0.02),
            a_t=aggregates.get("gdp_acceleration", 0.0),
            theta_t=aggregates.get("sectoral_coherence", 0.7),
            t_adjusted=tensions.get("t_adjusted", 0.3),
            t_f=tensions.get("t_financial", 0.2),
            t_e=tensions.get("t_energy", 0.2),
            m_macro=self.memory_system.state.macro_memory,
            capacity_utilization=0.85,  # Placeholder
            duration_in_phase=self.phase_engine.time_in_current_phase,
        )

        # Force crisis if black swan event
        force_crisis = (
            event is not None
            and event.event_type == EventType.BLACK_SWAN
            and event.effective_impact < -0.1
        )

        new_phase, probs = self.phase_engine.transition(
            self.current_phase,
            conditions,
            force_crisis=force_crisis,
        )

        return new_phase

    def _step_learning_evolution(
        self,
        all_actions: Dict[int, Dict[str, float]],
        aggregates: Dict[str, float],
    ) -> None:
        """
        Step 5: Learning and network evolution.

        Update agent policies and potentially rewire network.
        """
        # Compute rewards for each agent
        agent_rewards = {}

        for agent_id, household in self.households.items():
            wealth_change = household.state.wealth - household.state.income
            agent_rewards[agent_id] = wealth_change

        for agent_id, firm in self.firms.items():
            agent_rewards[agent_id] = firm.state.profits

        for agent_id, bank in self.banks.items():
            agent_rewards[agent_id] = bank.state.capital - bank.state.loans * bank.state.npl_ratio

        # Update memory system
        sector_shocks = {
            sector: (aggregates["sector_outputs"].get(sector, 0) - aggregates["gdp"] / 5)
            for sector in ["primary", "manufacturing", "services", "financial", "public"]
        }

        systemic_impact = 0.0
        if self.results.events and self.results.events[-1]:
            last_event = self.results.events[-1]
            systemic_impact = last_event.get("effective_impact", 0)

        self.memory_system.update_all(
            agent_rewards=agent_rewards,
            sector_shocks=sector_shocks,
            systemic_event_impact=systemic_impact,
        )

        # Network rewiring (optional, based on agent performance)
        agent_states = {}
        for agent_id, household in self.households.items():
            agent_states[agent_id] = {
                "wealth": household.state.wealth,
                "is_bankrupt": False,
            }
        for agent_id, firm in self.firms.items():
            agent_states[agent_id] = {
                "wealth": firm.state.wealth,
                "is_bankrupt": firm.state.is_bankrupt,
            }

        self.network.rewire_adaptive(agent_states, rewire_prob=0.01)

    def _update_agents(
        self,
        all_actions: Dict[int, Dict[str, float]],
        aggregates: Dict[str, float],
    ) -> None:
        """Update all agents after market clearing."""
        market_outcomes = {
            "credit_ratio": min(1, aggregates["credit_supply"] / max(1, aggregates["credit_demand"])),
            "demand_ratio": 1.0,  # Simplified
        }

        for agent_id, household in self.households.items():
            household.update(
                all_actions[agent_id],
                self.macro_state,
                market_outcomes,
            )

        for agent_id, firm in self.firms.items():
            firm.update(
                all_actions[agent_id],
                self.macro_state,
                market_outcomes,
            )

        for agent_id, bank in self.banks.items():
            bank.update(
                all_actions[agent_id],
                self.macro_state,
                market_outcomes,
            )

        self.government.update(
            all_actions[self.government.agent_id],
            self.macro_state,
            {"unemployment": aggregates["unemployment"], "inflation": aggregates["inflation"]},
        )

    def _update_macro_state(
        self,
        aggregates: Dict[str, float],
        tensions: Dict[str, float],
        new_phase: EconomicPhase,
    ) -> None:
        """Update the macro state object."""
        self.macro_state.phase = new_phase.value
        self.macro_state.gdp = aggregates["gdp"]
        self.macro_state.gdp_growth = aggregates["gdp_growth"]
        self.macro_state.gdp_acceleration = aggregates["gdp_acceleration"]
        self.macro_state.sectoral_coherence = aggregates["sectoral_coherence"]
        self.macro_state.unemployment = aggregates["unemployment"]
        self.macro_state.inflation = aggregates["inflation"]
        self.macro_state.interest_rate = self.government.state.policy_rate

        self.macro_state.tensions = np.array([
            tensions.get("t_energy", 0.2),
            tensions.get("t_trade", 0.2),
            tensions.get("t_currency", 0.2),
            tensions.get("t_financial", 0.2),
            tensions.get("t_events", 0.0),
        ])

        self.macro_state.memory = self.memory_system.get_memory_vector()

        self.current_phase = new_phase

    def _record_results(
        self,
        aggregates: Dict[str, float],
        tensions: Dict[str, float],
        event: Optional[Any],
        policy: Dict[str, float],
    ) -> None:
        """Record results for this step."""
        self.results.gdp.append(aggregates["gdp"])
        self.results.gdp_growth.append(aggregates["gdp_growth"])
        self.results.unemployment.append(aggregates["unemployment"])
        self.results.inflation.append(aggregates["inflation"])

        self.results.phases.append(self.current_phase.value)

        self.results.tensions.append(tensions)

        if event:
            self.results.events.append({
                "type": event.event_type.value,
                "magnitude": event.magnitude,
                "effective_impact": event.effective_impact,
                "tension_target": event.tension_target,
            })
        else:
            self.results.events.append(None)

        self.results.memory.append(self.memory_system.to_dict())
        self.results.policy.append(policy)

    def _get_agent(self, agent_id: int):
        """Get agent by ID."""
        if agent_id in self.households:
            return self.households[agent_id]
        if agent_id in self.firms:
            return self.firms[agent_id]
        if agent_id in self.banks:
            return self.banks[agent_id]
        if agent_id == self.government.agent_id:
            return self.government
        return None

    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.

        Implements full Algorithm 2 iteration.
        """
        # Step 1: Local interaction
        all_actions = self._step_local_interaction()

        # Step 2: Aggregation
        aggregates = self._step_aggregation(all_actions)

        # Step 3: Events and tensions
        tensions, event = self._step_events_tensions(aggregates)

        # Step 4: Phase transition
        new_phase = self._step_phase_transition(aggregates, tensions, event)

        # Apply MMT stabilizers
        output_gap = (aggregates["gdp"] - self.macro_state.potential_output) / max(
            self.macro_state.potential_output, 0.01
        )
        policy = self.mmt.stabilize(
            output_gap=output_gap,
            t_adjusted=tensions.get("t_adjusted", 0.3),
            t_energy=tensions.get("t_energy", 0.2),
            t_trade=tensions.get("t_trade", 0.2),
            capacity_utilization=0.85,
            private_employment=1 - aggregates["unemployment"],
            is_crisis=new_phase == EconomicPhase.CRISIS,
        )

        # Update macro state
        self._update_macro_state(aggregates, tensions, new_phase)

        # Update agents
        self._update_agents(all_actions, aggregates)

        # Step 5: Learning and evolution
        self._step_learning_evolution(all_actions, aggregates)

        # Record results
        self._record_results(aggregates, tensions, event, policy)

        self.current_step += 1

        return {
            "step": self.current_step,
            "phase": self.current_phase.value,
            "gdp": aggregates["gdp"],
            "gdp_growth": aggregates["gdp_growth"],
            "unemployment": aggregates["unemployment"],
            "inflation": aggregates["inflation"],
            "tension": tensions.get("t_adjusted", 0.3),
            "event": event.event_type.value if event else None,
        }

    def run(
        self,
        n_steps: Optional[int] = None,
        progress_bar: bool = True,
    ) -> SimulationResults:
        """
        Run the full simulation.

        Args:
            n_steps: Number of steps to run (defaults to params.time_horizon)
            progress_bar: Show progress bar

        Returns:
            SimulationResults object
        """
        n_steps = n_steps or self.params.time_horizon

        iterator = range(n_steps)
        if progress_bar:
            iterator = tqdm(iterator, desc="Simulating")

        for _ in iterator:
            self.step()

        # Compute phase durations
        for phase in EconomicPhase:
            self.results.phase_durations[phase.value] = sum(
                1 for p in self.results.phases if p == phase.value
            )

        return self.results

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.current_phase = EconomicPhase.ACTIVATION
        self.current_step = 0
        self.macro_state = MacroState()
        self.results = SimulationResults()

        # Reset subsystems
        self.phase_engine.reset()
        self.gdp_vector.reset()
        self.tension_system.reset()
        self.event_generator.reset()
        self.memory_system.reset()

        # Reset agents
        for household in self.households.values():
            household.reset()
        for firm in self.firms.values():
            firm.reset()
        for bank in self.banks.values():
            bank.reset()
        self.government.reset()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the simulation."""
        if not self.results.gdp:
            return {"status": "not_run"}

        return {
            "steps": len(self.results.gdp),
            "final_phase": self.results.phases[-1] if self.results.phases else None,
            "avg_gdp_growth": np.mean(self.results.gdp_growth),
            "std_gdp_growth": np.std(self.results.gdp_growth),
            "avg_unemployment": np.mean(self.results.unemployment),
            "avg_inflation": np.mean(self.results.inflation),
            "n_crises": sum(1 for p in self.results.phases if p == "crisis"),
            "n_events": sum(1 for e in self.results.events if e is not None),
            "phase_durations": self.results.phase_durations,
        }
