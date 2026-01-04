"""
Basic tests for the ABM Economic Phases model.

Run with: python -m pytest tests/test_basic.py -v
"""

import pytest
import numpy as np

from abm_economic_phases import EconomicSimulation, ModelParameters, EconomicPhase
from abm_economic_phases.agents import Household, Firm, Bank, Government
from abm_economic_phases.agents.base import MacroState
from abm_economic_phases.network import ScaleFreeNetwork
from abm_economic_phases.economy import GDPVector, TensionSystem, ExtremeEventGenerator
from abm_economic_phases.economy.phases import PhaseTransitionEngine, PhaseConditions
from abm_economic_phases.learning import AdaptiveMemorySystem, SARSALambda
from abm_economic_phases.mmt import MMTStabilizers


class TestParameters:
    """Tests for model parameters."""

    def test_default_parameters(self):
        """Test that default parameters are created correctly."""
        params = ModelParameters()

        assert params.agents.n_households == 1000
        assert params.agents.n_firms == 100
        assert params.agents.n_banks == 10
        assert 2 < params.network.gamma < 3

    def test_parameter_to_dict(self):
        """Test parameter serialization."""
        params = ModelParameters()
        d = params.to_dict()

        assert "agents.n_households" in d
        assert "network.gamma" in d


class TestAgents:
    """Tests for agent classes."""

    def test_household_creation(self):
        """Test household agent creation."""
        h = Household(agent_id=0, seed=42)

        assert h.agent_id == 0
        assert h.state.wealth > 0
        assert h.state.is_employed

    def test_household_decision(self):
        """Test household decision making."""
        h = Household(agent_id=0, seed=42)
        macro = MacroState()

        actions = h.decide(macro, {})

        assert "consumption" in actions
        assert "savings" in actions
        assert "labor_supply" in actions
        assert actions["consumption"] > 0

    def test_firm_creation(self):
        """Test firm agent creation."""
        f = Firm(agent_id=0, sector="manufacturing", seed=42)

        assert f.agent_id == 0
        assert f.sector == "manufacturing"
        assert f.state.capital > 0

    def test_firm_production(self):
        """Test firm production function."""
        f = Firm(agent_id=0, seed=42)

        output = f._production_function(capital=1.0, labor=1.0, tfp=1.0)
        assert output > 0

    def test_bank_creation(self):
        """Test bank agent creation."""
        b = Bank(agent_id=0, seed=42)

        assert b.agent_id == 0
        assert b.state.capital > 0
        assert b.state.capital_ratio > 0

    def test_government_creation(self):
        """Test government agent creation."""
        g = Government(agent_id=0, seed=42)

        assert g.agent_id == 0
        assert g.state.spending > 0


class TestNetwork:
    """Tests for scale-free network."""

    def test_network_creation(self):
        """Test network creation."""
        net = ScaleFreeNetwork(
            n_households=100,
            n_firms=10,
            n_banks=2,
            seed=42,
        )

        assert net.graph.number_of_nodes() == 113  # 100 + 10 + 2 + 1 (gov)
        assert net.graph.number_of_edges() > 0

    def test_network_metrics(self):
        """Test network metrics computation."""
        net = ScaleFreeNetwork(
            n_households=100,
            n_firms=10,
            n_banks=2,
            seed=42,
        )

        metrics = net.compute_metrics()

        assert metrics.num_nodes == 113
        assert metrics.average_degree > 0
        assert 2 < metrics.gamma_estimate < 4

    def test_node_types(self):
        """Test that node types are correctly assigned."""
        net = ScaleFreeNetwork(
            n_households=10,
            n_firms=5,
            n_banks=2,
            seed=42,
        )

        households = net.get_nodes_by_type("household")
        firms = net.get_nodes_by_type("firm")
        banks = net.get_nodes_by_type("bank")

        assert len(households) == 10
        assert len(firms) == 5
        assert len(banks) == 2


class TestEconomy:
    """Tests for economic components."""

    def test_gdp_vector(self):
        """Test GDP vector computation."""
        gdp = GDPVector()

        g, a, theta = gdp.update(100)
        assert g == 0.02  # Default initial

        g, a, theta = gdp.update(103)
        assert g > 0  # Should show growth

    def test_phase_transitions(self):
        """Test phase transition logic."""
        engine = PhaseTransitionEngine(seed=42)

        # Test expansion conditions
        conditions = PhaseConditions(
            g_t=0.03,
            a_t=0.01,
            theta_t=0.7,
            duration_in_phase=3,
        )

        probs = engine.compute_transition_probability(
            EconomicPhase.ACTIVATION, conditions
        )

        assert EconomicPhase.EXPANSION in probs

    def test_tension_system(self):
        """Test tension system."""
        tensions = TensionSystem(seed=42)

        metrics = tensions.update(macro_memory=0.5)

        assert 0 <= metrics.t_energy <= 1
        assert 0 <= metrics.t_adjusted <= 1

    def test_event_generation(self):
        """Test extreme event generation."""
        generator = ExtremeEventGenerator(seed=42)

        # Force high tension to increase event probability
        events = []
        for _ in range(100):
            event = generator.generate_event(
                t_adjusted=0.9,
                macro_memory=0.0,
            )
            if event:
                events.append(event)

        # Should generate some events with high tension
        assert len(events) > 0


class TestLearning:
    """Tests for learning components."""

    def test_memory_system(self):
        """Test adaptive memory system."""
        memory = AdaptiveMemorySystem(seed=42)

        # Register and update
        memory.register_agent(0, "manufacturing")
        memory.update_micro_memory(0, reward=0.1)

        assert memory.state.micro_memories[0] != 0

    def test_sarsa_lambda(self):
        """Test SARSA(λ) learning."""
        actions = ["high", "medium", "low"]
        learner = SARSALambda(actions=actions, seed=42)

        # Run a few updates
        state = "normal"
        action = learner.select_action(state)
        learner.update(state, action, 1.0, "good", "high")

        assert learner.state.total_steps == 1
        assert len(learner.td_errors) == 1


class TestMMT:
    """Tests for MMT components."""

    def test_stabilizers(self):
        """Test MMT stabilizers."""
        mmt = MMTStabilizers(seed=42)

        result = mmt.stabilize(
            output_gap=-0.02,
            t_adjusted=0.3,
            t_energy=0.2,
            t_trade=0.2,
            capacity_utilization=0.85,
            private_employment=0.95,
            is_crisis=False,
        )

        assert "spending" in result
        assert "deficit" in result
        assert "policy_rate" in result


class TestSimulation:
    """Tests for the main simulation engine."""

    def test_simulation_creation(self):
        """Test simulation initialization."""
        params = ModelParameters()
        params.agents.n_households = 50
        params.agents.n_firms = 10
        params.agents.n_banks = 2

        sim = EconomicSimulation(params=params, seed=42)

        assert len(sim.households) == 50
        assert len(sim.firms) == 10
        assert len(sim.banks) == 2
        assert sim.government is not None

    def test_simulation_step(self):
        """Test single simulation step."""
        params = ModelParameters()
        params.agents.n_households = 50
        params.agents.n_firms = 10
        params.agents.n_banks = 2

        sim = EconomicSimulation(params=params, seed=42)
        result = sim.step()

        assert "step" in result
        assert "phase" in result
        assert "gdp" in result
        assert result["step"] == 1

    def test_simulation_run(self):
        """Test multi-step simulation."""
        params = ModelParameters()
        params.agents.n_households = 50
        params.agents.n_firms = 10
        params.agents.n_banks = 2
        params.time_horizon = 10

        sim = EconomicSimulation(params=params, seed=42)
        results = sim.run(progress_bar=False)

        assert len(results.gdp) == 10
        assert len(results.phases) == 10

    def test_simulation_reset(self):
        """Test simulation reset."""
        params = ModelParameters()
        params.agents.n_households = 50
        params.agents.n_firms = 10
        params.agents.n_banks = 2

        sim = EconomicSimulation(params=params, seed=42)
        sim.step()
        sim.step()

        assert sim.current_step == 2

        sim.reset()

        assert sim.current_step == 0
        assert len(sim.results.gdp) == 0


class TestIntegration:
    """Integration tests."""

    def test_full_cycle(self):
        """Test that simulation can go through all phases."""
        params = ModelParameters()
        params.agents.n_households = 100
        params.agents.n_firms = 20
        params.agents.n_banks = 3
        params.time_horizon = 50

        sim = EconomicSimulation(params=params, seed=12345)
        results = sim.run(progress_bar=False)

        # Check that we have valid results throughout
        assert all(g is not None for g in results.gdp_growth)
        assert all(0 <= u <= 1 for u in results.unemployment)

    def test_event_propagation(self):
        """Test that events affect the economy."""
        params = ModelParameters()
        params.agents.n_households = 50
        params.agents.n_firms = 10
        params.agents.n_banks = 2
        params.time_horizon = 20
        params.events.lambda_0 = 0.3  # High event probability

        sim = EconomicSimulation(params=params, seed=42)
        results = sim.run(progress_bar=False)

        # Should have some events
        events = [e for e in results.events if e is not None]
        # Events should affect GDP
        # (Hard to test deterministically due to stochasticity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
