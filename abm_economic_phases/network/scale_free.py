"""
Scale-Free Network Topology

Implements the preferential attachment mechanism from Section 3.2:
P(connection to i) = (k_i + k_0) / sum_j(k_j + k_0)

Resulting distribution follows: P(k) ~ k^{-gamma}, with 2 < gamma < 3

Key systemic properties:
- Robustness to random failures: R_rand ≈ 1 - exp(-<k>)
- Fragility to targeted attacks: R_targeted ≈ exp(-k_max/<k>)
- Accelerated diffusion: tau_diffusion ~ log(N) / log(log(N))
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import networkx as nx
from enum import Enum


class AgentLayer(Enum):
    """Network layers for different agent types."""

    HOUSEHOLD = "household"
    FIRM = "firm"
    BANK = "bank"
    INTER_TYPE = "inter_type"  # Cross-type connections


@dataclass
class NetworkMetrics:
    """Metrics for network analysis."""

    # Basic statistics
    num_nodes: int = 0
    num_edges: int = 0
    average_degree: float = 0.0
    max_degree: int = 0

    # Distribution parameters
    gamma_estimate: float = 2.5  # Power-law exponent

    # Systemic properties
    robustness_random: float = 0.0
    robustness_targeted: float = 0.0
    diffusion_time: float = 0.0

    # Centrality measures
    avg_clustering: float = 0.0
    avg_path_length: float = 0.0
    diameter: int = 0

    # Hub identification
    hub_nodes: List[int] = None

    def __post_init__(self):
        if self.hub_nodes is None:
            self.hub_nodes = []


class ScaleFreeNetwork:
    """
    Scale-free network generator and manager using preferential attachment.

    Implements Barabási-Albert model with modifications for economic networks:
    - Multi-layer structure (households, firms, banks)
    - Intrinsic attractiveness parameter k_0
    - Dynamic rewiring capabilities
    """

    def __init__(
        self,
        n_households: int = 1000,
        n_firms: int = 100,
        n_banks: int = 10,
        m: int = 3,  # Edges per new node
        k_0: float = 1.0,  # Intrinsic attraction
        seed: Optional[int] = None,
    ):
        self.n_households = n_households
        self.n_firms = n_firms
        self.n_banks = n_banks
        self.m = m
        self.k_0 = k_0
        self.rng = np.random.default_rng(seed)

        # Initialize multi-layer graph
        self.graph = nx.Graph()
        self._node_types: Dict[int, str] = {}
        self._type_nodes: Dict[str, Set[int]] = {
            "household": set(),
            "firm": set(),
            "bank": set(),
            "government": set(),
        }

        # Build the network
        self._build_network()

    def _build_network(self) -> None:
        """Build the complete multi-layer scale-free network."""
        # Start with banks (core of financial network)
        self._add_bank_layer()

        # Add firms connected to banks
        self._add_firm_layer()

        # Add households connected to firms and banks
        self._add_household_layer()

        # Add government node (connected to all)
        self._add_government_node()

    def _add_bank_layer(self) -> None:
        """Create the interbank network layer."""
        # Banks form a dense core
        bank_ids = list(range(self.n_banks))

        for bank_id in bank_ids:
            self.graph.add_node(bank_id, type="bank")
            self._node_types[bank_id] = "bank"
            self._type_nodes["bank"].add(bank_id)

        # Connect banks (nearly complete graph for interbank market)
        for i in range(len(bank_ids)):
            for j in range(i + 1, len(bank_ids)):
                if self.rng.random() < 0.7:  # 70% connection probability
                    self.graph.add_edge(bank_ids[i], bank_ids[j], layer="bank")

    def _add_firm_layer(self) -> None:
        """Add firms with preferential attachment to banks."""
        start_id = self.n_banks

        for i in range(self.n_firms):
            firm_id = start_id + i
            self.graph.add_node(firm_id, type="firm")
            self._node_types[firm_id] = "firm"
            self._type_nodes["firm"].add(firm_id)

            # Connect to banks (preferential attachment)
            bank_connections = self._select_targets(
                list(self._type_nodes["bank"]),
                min(self.m, self.n_banks),
            )
            for bank_id in bank_connections:
                self.graph.add_edge(firm_id, bank_id, layer="firm_bank")

            # Connect to other firms (preferential attachment)
            if i > self.m:
                existing_firms = [fid for fid in self._type_nodes["firm"] if fid < firm_id]
                firm_connections = self._select_targets(
                    existing_firms,
                    min(self.m, len(existing_firms)),
                )
                for other_firm in firm_connections:
                    self.graph.add_edge(firm_id, other_firm, layer="firm")

    def _add_household_layer(self) -> None:
        """Add households with preferential attachment to firms."""
        start_id = self.n_banks + self.n_firms

        for i in range(self.n_households):
            household_id = start_id + i
            self.graph.add_node(household_id, type="household")
            self._node_types[household_id] = "household"
            self._type_nodes["household"].add(household_id)

            # Connect to firms (employer relationship)
            firm_connections = self._select_targets(
                list(self._type_nodes["firm"]),
                min(self.m, self.n_firms),
            )
            for firm_id in firm_connections:
                self.graph.add_edge(household_id, firm_id, layer="household_firm")

            # Connect to bank (banking relationship)
            bank_connections = self._select_targets(
                list(self._type_nodes["bank"]),
                1,  # One primary bank
            )
            for bank_id in bank_connections:
                self.graph.add_edge(household_id, bank_id, layer="household_bank")

            # Connect to other households (social network)
            if i > self.m:
                existing_households = [
                    hid for hid in self._type_nodes["household"] if hid < household_id
                ]
                if existing_households:
                    household_connections = self._select_targets(
                        existing_households,
                        min(2, len(existing_households)),
                    )
                    for other_hh in household_connections:
                        self.graph.add_edge(
                            household_id, other_hh, layer="household"
                        )

    def _add_government_node(self) -> None:
        """Add government node connected to key hubs."""
        gov_id = self.n_banks + self.n_firms + self.n_households
        self.graph.add_node(gov_id, type="government")
        self._node_types[gov_id] = "government"
        self._type_nodes["government"].add(gov_id)

        # Connect to all banks
        for bank_id in self._type_nodes["bank"]:
            self.graph.add_edge(gov_id, bank_id, layer="government_bank")

        # Connect to top firms by degree
        firm_degrees = [
            (fid, self.graph.degree(fid)) for fid in self._type_nodes["firm"]
        ]
        firm_degrees.sort(key=lambda x: x[1], reverse=True)
        top_firms = [fid for fid, _ in firm_degrees[: self.n_firms // 10]]

        for firm_id in top_firms:
            self.graph.add_edge(gov_id, firm_id, layer="government_firm")

    def _select_targets(
        self,
        candidates: List[int],
        num_targets: int,
    ) -> List[int]:
        """
        Select connection targets using preferential attachment.

        P(connect to i) = (k_i + k_0) / sum_j(k_j + k_0)
        """
        if not candidates or num_targets <= 0:
            return []

        if len(candidates) <= num_targets:
            return candidates

        # Calculate attachment probabilities
        degrees = np.array([self.graph.degree(node) for node in candidates])
        weights = degrees + self.k_0
        probabilities = weights / weights.sum()

        # Sample without replacement
        selected_indices = self.rng.choice(
            len(candidates),
            size=min(num_targets, len(candidates)),
            replace=False,
            p=probabilities,
        )

        return [candidates[i] for i in selected_indices]

    def get_neighbors(self, node_id: int) -> Set[int]:
        """Get all neighbors of a node."""
        if node_id in self.graph:
            return set(self.graph.neighbors(node_id))
        return set()

    def get_neighbors_by_type(
        self,
        node_id: int,
        node_type: str,
    ) -> Set[int]:
        """Get neighbors of a specific type."""
        neighbors = self.get_neighbors(node_id)
        return {n for n in neighbors if self._node_types.get(n) == node_type}

    def get_node_type(self, node_id: int) -> Optional[str]:
        """Get the type of a node."""
        return self._node_types.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> Set[int]:
        """Get all nodes of a specific type."""
        return self._type_nodes.get(node_type, set()).copy()

    def compute_metrics(self) -> NetworkMetrics:
        """Compute network metrics including systemic properties."""
        degrees = [d for _, d in self.graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        metrics = NetworkMetrics(
            num_nodes=self.graph.number_of_nodes(),
            num_edges=self.graph.number_of_edges(),
            average_degree=avg_degree,
            max_degree=max_degree,
        )

        # Estimate power-law exponent gamma
        if len(degrees) > 10:
            metrics.gamma_estimate = self._estimate_gamma(degrees)

        # Systemic properties from paper
        # R_rand ≈ 1 - exp(-<k>)
        metrics.robustness_random = 1 - np.exp(-avg_degree)

        # R_targeted ≈ exp(-k_max / <k>)
        if avg_degree > 0:
            metrics.robustness_targeted = np.exp(-max_degree / avg_degree)
        else:
            metrics.robustness_targeted = 0.0

        # tau_diffusion ~ log(N) / log(log(N))
        n = self.graph.number_of_nodes()
        if n > 10:
            metrics.diffusion_time = np.log(n) / max(1, np.log(np.log(n)))
        else:
            metrics.diffusion_time = n

        # Clustering coefficient
        metrics.avg_clustering = nx.average_clustering(self.graph)

        # Path length (sample for large networks)
        if n < 500:
            try:
                if nx.is_connected(self.graph):
                    metrics.avg_path_length = nx.average_shortest_path_length(
                        self.graph
                    )
                    metrics.diameter = nx.diameter(self.graph)
            except nx.NetworkXError:
                pass

        # Identify hubs (top 5% by degree)
        sorted_nodes = sorted(
            self.graph.degree(), key=lambda x: x[1], reverse=True
        )
        n_hubs = max(1, len(sorted_nodes) // 20)
        metrics.hub_nodes = [node for node, _ in sorted_nodes[:n_hubs]]

        return metrics

    def _estimate_gamma(self, degrees: List[int]) -> float:
        """
        Estimate power-law exponent using maximum likelihood.

        For P(k) ~ k^{-gamma}, MLE gives:
        gamma = 1 + n / sum(ln(k_i / k_min))
        """
        k_min = max(1, min(degrees))
        filtered = [k for k in degrees if k >= k_min]

        if len(filtered) < 10:
            return 2.5  # Default

        n = len(filtered)
        log_sum = sum(np.log(k / k_min) for k in filtered if k > k_min)

        if log_sum > 0:
            gamma = 1 + n / log_sum
            return np.clip(gamma, 2.0, 3.5)

        return 2.5

    def rewire_adaptive(
        self,
        agent_states: Dict[int, Any],
        rewire_prob: float = 0.01,
    ) -> int:
        """
        Adaptive rewiring based on agent performance.

        Agents tend to disconnect from failed/bankrupt neighbors
        and reconnect to successful ones.

        Returns number of rewiring events.
        """
        rewiring_count = 0

        for node_id in list(self.graph.nodes()):
            if self.rng.random() > rewire_prob:
                continue

            node_type = self._node_types.get(node_id)
            if node_type == "government":
                continue

            neighbors = list(self.graph.neighbors(node_id))
            if not neighbors:
                continue

            # Evaluate neighbors
            neighbor_scores = []
            for neighbor in neighbors:
                state = agent_states.get(neighbor, {})
                score = state.get("wealth", 1.0)
                if state.get("is_bankrupt", False):
                    score = -1.0
                neighbor_scores.append((neighbor, score))

            # Consider dropping worst connection
            neighbor_scores.sort(key=lambda x: x[1])
            worst_neighbor, worst_score = neighbor_scores[0]

            if worst_score < 0 or (
                len(neighbors) > 2 and worst_score < 0.5 * np.mean([s for _, s in neighbor_scores])
            ):
                # Find better connection
                same_type_nodes = self._type_nodes.get(node_type, set())
                non_neighbors = same_type_nodes - set(neighbors) - {node_id}

                if non_neighbors:
                    # Preferential attachment to high-scoring nodes
                    candidates = list(non_neighbors)
                    scores = [
                        max(0.01, agent_states.get(c, {}).get("wealth", 1.0))
                        for c in candidates
                    ]
                    probs = np.array(scores) / sum(scores)

                    new_neighbor = self.rng.choice(candidates, p=probs)

                    # Rewire
                    self.graph.remove_edge(node_id, worst_neighbor)
                    self.graph.add_edge(node_id, new_neighbor, layer="adaptive")
                    rewiring_count += 1

        return rewiring_count

    def simulate_contagion(
        self,
        initial_infected: Set[int],
        infection_prob: float = 0.5,
        recovery_prob: float = 0.1,
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Simulate contagion (e.g., financial distress, information) on the network.

        SIR-like model for studying systemic risk propagation.
        """
        infected = initial_infected.copy()
        recovered = set()
        history = [len(infected)]

        for step in range(max_steps):
            new_infected = set()

            for node in infected:
                # Spread to neighbors
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in infected and neighbor not in recovered:
                        if self.rng.random() < infection_prob:
                            new_infected.add(neighbor)

            # Recovery
            for node in list(infected):
                if self.rng.random() < recovery_prob:
                    infected.remove(node)
                    recovered.add(node)

            infected.update(new_infected)
            history.append(len(infected))

            if not infected:
                break

        return {
            "peak_infected": max(history),
            "total_infected": len(recovered) + len(infected),
            "duration": len(history),
            "history": history,
        }

    def targeted_attack(
        self,
        fraction: float = 0.05,
    ) -> Tuple[nx.Graph, List[int]]:
        """
        Simulate targeted attack on highest-degree nodes.

        Returns the remaining graph and list of removed nodes.
        """
        degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
        n_remove = max(1, int(fraction * len(degrees)))
        removed = [node for node, _ in degrees[:n_remove]]

        remaining = self.graph.copy()
        remaining.remove_nodes_from(removed)

        return remaining, removed

    def random_failure(
        self,
        fraction: float = 0.05,
    ) -> Tuple[nx.Graph, List[int]]:
        """
        Simulate random node failures.

        Returns the remaining graph and list of removed nodes.
        """
        all_nodes = list(self.graph.nodes())
        n_remove = max(1, int(fraction * len(all_nodes)))
        removed = list(self.rng.choice(all_nodes, size=n_remove, replace=False))

        remaining = self.graph.copy()
        remaining.remove_nodes_from(removed)

        return remaining, removed

    def to_dict(self) -> Dict[str, Any]:
        """Export network to dictionary format."""
        return {
            "nodes": [
                {"id": n, "type": self._node_types.get(n)}
                for n in self.graph.nodes()
            ],
            "edges": [
                {"source": u, "target": v, "layer": d.get("layer", "default")}
                for u, v, d in self.graph.edges(data=True)
            ],
            "metrics": self.compute_metrics().__dict__,
        }
