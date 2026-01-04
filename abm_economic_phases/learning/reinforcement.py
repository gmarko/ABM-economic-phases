"""
Reinforcement Learning Module

Implements Algorithm 1: SARSA(λ) with eligibility traces and memory incorporation.

Key update rules:
1. TD error: δ_t = R_t + γ*Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
2. Trace update: e(s,a) ← e(s,a) + 1
3. Q update: Q(s,a) ← Q(s,a) + α*δ_t*e(s,a)
4. Trace decay: e(s,a) ← γ*λ*e(s,a)
5. Policy: π(a|s) = exp(β*Q(s,a)) / Σ_a' exp(β*Q(s,a'))
6. Memory incorporation: π ← (1-η)*π + η*softmax(M_micro)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class LearningState:
    """State for a learning agent."""

    # Q-values: mapping (state, action) -> value
    q_values: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Eligibility traces: same structure as Q
    traces: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Current policy: mapping action -> probability (for current state)
    policy: Dict[str, float] = field(default_factory=dict)

    # Experience
    total_episodes: int = 0
    total_steps: int = 0


class SARSALambda:
    """
    SARSA(λ) learning algorithm with eligibility traces.

    Implements on-policy TD learning suitable for continuous
    adaptation in the economic environment.
    """

    def __init__(
        self,
        actions: List[str],
        alpha: float = 0.1,  # Learning rate
        gamma: float = 0.95,  # Discount factor
        lambda_trace: float = 0.8,  # Eligibility trace decay
        beta_softmax: float = 5.0,  # Softmax temperature
        epsilon_start: float = 0.3,  # Initial exploration
        epsilon_end: float = 0.05,  # Final exploration
        epsilon_decay: float = 0.995,  # Exploration decay
        seed: Optional[int] = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.beta_softmax = beta_softmax
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Learning state
        self.state = LearningState()

        # History for analysis
        self.td_errors: List[float] = []
        self.rewards: List[float] = []

    def get_q_value(
        self,
        state: str,
        action: str,
    ) -> float:
        """Get Q-value for state-action pair, with optimistic initialization."""
        return self.state.q_values.get((state, action), 0.0)

    def get_trace(
        self,
        state: str,
        action: str,
    ) -> float:
        """Get eligibility trace for state-action pair."""
        return self.state.traces.get((state, action), 0.0)

    def select_action(
        self,
        state: str,
        greedy: bool = False,
    ) -> str:
        """
        Select action using epsilon-greedy with softmax.

        With probability epsilon: random action
        Otherwise: sample from softmax policy
        """
        if not greedy and self.rng.random() < self.epsilon:
            # Exploration
            return self.rng.choice(self.actions)

        # Compute softmax policy
        q_values = [self.get_q_value(state, a) for a in self.actions]
        max_q = max(q_values) if q_values else 0

        # Numerical stability
        exp_values = [np.exp(self.beta_softmax * (q - max_q)) for q in q_values]
        total = sum(exp_values)

        if total > 0:
            probs = [e / total for e in exp_values]
        else:
            probs = [1 / len(self.actions)] * len(self.actions)

        # Store policy
        self.state.policy = dict(zip(self.actions, probs))

        # Sample from policy
        return self.rng.choice(self.actions, p=probs)

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_action: str,
    ) -> float:
        """
        Perform SARSA(λ) update.

        Returns TD error for monitoring.
        """
        # Step 3: Compute TD error
        current_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action)
        td_error = reward + self.gamma * next_q - current_q

        # Step 4: Update trace for current state-action
        current_trace = self.get_trace(state, action)
        self.state.traces[(state, action)] = current_trace + 1

        # Steps 5-7: Update all Q-values and decay traces
        for (s, a), trace in list(self.state.traces.items()):
            if trace > 0.001:  # Threshold for efficiency
                # Q update
                old_q = self.get_q_value(s, a)
                new_q = old_q + self.alpha * td_error * trace
                self.state.q_values[(s, a)] = new_q

                # Trace decay
                self.state.traces[(s, a)] = self.gamma * self.lambda_trace * trace
            else:
                del self.state.traces[(s, a)]

        # Track statistics
        self.td_errors.append(td_error)
        self.rewards.append(reward)
        self.state.total_steps += 1

        # Decay exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return td_error

    def incorporate_memory(
        self,
        memory_value: float,
        eta: float = 0.1,
    ) -> None:
        """
        Incorporate individual memory into policy (Step 9 of Algorithm 1).

        π ← (1-η)*π + η*softmax(M_micro)

        Memory provides a prior that biases action selection.
        """
        if not self.state.policy:
            return

        # Memory-based action preferences
        # Positive memory → favor conservative actions
        # Negative memory → favor aggressive actions
        memory_prefs = {}
        for action in self.actions:
            if "conservative" in action or "save" in action:
                memory_prefs[action] = memory_value
            elif "aggressive" in action or "invest" in action:
                memory_prefs[action] = -memory_value
            else:
                memory_prefs[action] = 0.0

        # Softmax of memory preferences
        exp_prefs = {a: np.exp(p) for a, p in memory_prefs.items()}
        total = sum(exp_prefs.values())
        memory_policy = {a: e / total for a, e in exp_prefs.items()}

        # Blend with current policy
        for action in self.actions:
            current = self.state.policy.get(action, 1 / len(self.actions))
            memory = memory_policy.get(action, 1 / len(self.actions))
            self.state.policy[action] = (1 - eta) * current + eta * memory

    def reset_traces(self) -> None:
        """Reset eligibility traces (start of new episode)."""
        self.state.traces.clear()
        self.state.total_episodes += 1

    def get_value_function(self, state: str) -> float:
        """Get state value V(s) = sum_a π(a|s) * Q(s,a)."""
        q_values = [self.get_q_value(state, a) for a in self.actions]
        probs = list(self.state.policy.values()) if self.state.policy else [1 / len(self.actions)] * len(self.actions)

        return sum(p * q for p, q in zip(probs, q_values))

    def get_statistics(self) -> Dict[str, float]:
        """Get learning statistics."""
        return {
            "avg_td_error": np.mean(self.td_errors[-100:]) if self.td_errors else 0,
            "avg_reward": np.mean(self.rewards[-100:]) if self.rewards else 0,
            "epsilon": self.epsilon,
            "num_states": len(set(s for s, _ in self.state.q_values.keys())),
            "total_steps": self.state.total_steps,
        }


class PolicyUpdater:
    """
    Policy update mechanism for the agent population.

    Handles:
    - Coordinated policy updates across agents
    - Policy diffusion through the network
    - Imitation learning from successful neighbors
    """

    def __init__(
        self,
        imitation_rate: float = 0.1,  # Rate of policy copying
        innovation_rate: float = 0.05,  # Rate of policy mutation
        seed: Optional[int] = None,
    ):
        self.imitation_rate = imitation_rate
        self.innovation_rate = innovation_rate
        self.rng = np.random.default_rng(seed)

    def imitate_neighbor(
        self,
        agent_learner: SARSALambda,
        neighbor_learner: SARSALambda,
        neighbor_performance: float,
        agent_performance: float,
    ) -> bool:
        """
        Potentially copy policy from a successful neighbor.

        Probability of imitation increases with performance gap.
        """
        if neighbor_performance <= agent_performance:
            return False

        # Performance-weighted imitation probability
        perf_gap = neighbor_performance - agent_performance
        imitation_prob = self.imitation_rate * np.tanh(perf_gap)

        if self.rng.random() > imitation_prob:
            return False

        # Copy Q-values (partial imitation)
        for key, value in neighbor_learner.state.q_values.items():
            if key in agent_learner.state.q_values:
                # Blend rather than replace
                agent_learner.state.q_values[key] = (
                    0.7 * agent_learner.state.q_values[key] + 0.3 * value
                )
            else:
                agent_learner.state.q_values[key] = value

        return True

    def innovate_policy(
        self,
        learner: SARSALambda,
        innovation_magnitude: float = 0.1,
    ) -> bool:
        """
        Add random innovation to policy (exploration/mutation).
        """
        if self.rng.random() > self.innovation_rate:
            return False

        # Add noise to Q-values
        for key in learner.state.q_values:
            noise = self.rng.normal(0, innovation_magnitude)
            learner.state.q_values[key] += noise

        return True

    def compute_policy_similarity(
        self,
        learner1: SARSALambda,
        learner2: SARSALambda,
    ) -> float:
        """
        Compute similarity between two agents' policies.

        Returns value in [0, 1] where 1 = identical policies.
        """
        # Find common state-action pairs
        keys1 = set(learner1.state.q_values.keys())
        keys2 = set(learner2.state.q_values.keys())
        common = keys1 & keys2

        if not common:
            return 0.5  # No common experience

        # Compute correlation of Q-values
        values1 = [learner1.state.q_values[k] for k in common]
        values2 = [learner2.state.q_values[k] for k in common]

        if len(values1) < 2:
            return 0.5

        correlation = np.corrcoef(values1, values2)[0, 1]
        if np.isnan(correlation):
            return 0.5

        # Map correlation [-1, 1] to similarity [0, 1]
        return 0.5 + 0.5 * correlation

    def aggregate_policies(
        self,
        learners: List[SARSALambda],
        weights: Optional[List[float]] = None,
    ) -> Dict[Tuple[str, str], float]:
        """
        Aggregate multiple agent policies into consensus policy.

        Useful for policy analysis or creating representative agent.
        """
        if not learners:
            return {}

        weights = weights or [1 / len(learners)] * len(learners)

        # Collect all state-action pairs
        all_keys = set()
        for learner in learners:
            all_keys.update(learner.state.q_values.keys())

        # Weighted average
        aggregate = {}
        for key in all_keys:
            values = []
            for learner, weight in zip(learners, weights):
                if key in learner.state.q_values:
                    values.append(weight * learner.state.q_values[key])

            if values:
                aggregate[key] = sum(values) / sum(
                    w for l, w in zip(learners, weights)
                    if key in l.state.q_values
                )

        return aggregate
