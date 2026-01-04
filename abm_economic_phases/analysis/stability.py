"""
Stability Analysis Module.

Implements Jacobian computation, eigenvalue analysis, and bifurcation detection
following the mathematical framework in Section 3 of the JEDC paper.

Key equations implemented:
- Eq. 6-7: Jacobian block structure
- Local stability via eigenvalue location
- Bifurcation detection from eigenvalue crossing unit circle
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable
from enum import Enum


class StabilityType(Enum):
    """Classification of stability at a fixed point."""
    STABLE = "stable"  # All eigenvalues inside unit circle
    UNSTABLE = "unstable"  # At least one eigenvalue outside unit circle
    SADDLE = "saddle"  # Eigenvalues on both sides of unit circle
    MARGINAL = "marginal"  # Eigenvalue(s) on unit circle (bifurcation)


@dataclass
class StabilityResult:
    """
    Result of stability analysis at a fixed point.

    Attributes:
        jacobian: The Jacobian matrix at the fixed point
        eigenvalues: Complex eigenvalues of the Jacobian
        eigenvectors: Corresponding eigenvectors
        stability_type: Classification of stability
        spectral_radius: Maximum absolute eigenvalue
        dominant_eigenvalue: Eigenvalue with largest modulus
        convergence_rate: Rate of convergence (if stable)
        oscillation_period: Period of oscillation (if complex eigenvalues)
    """
    jacobian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    stability_type: StabilityType
    spectral_radius: float
    dominant_eigenvalue: complex
    convergence_rate: Optional[float]
    oscillation_period: Optional[float]
    block_structure: Optional[Dict[str, np.ndarray]] = None


def compute_numerical_jacobian(
    transition_func: Callable,
    state: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute numerical Jacobian via central finite differences.

    J_ij = ∂F_i/∂S_j ≈ [F_i(S + ε*e_j) - F_i(S - ε*e_j)] / (2ε)

    Args:
        transition_func: Function F(S) returning next state
        state: Current state vector S_t
        epsilon: Perturbation size for finite differences

    Returns:
        Jacobian matrix J = ∂F/∂S
    """
    n = len(state)
    jacobian = np.zeros((n, n))

    for j in range(n):
        # Create perturbation vector
        perturbation = np.zeros(n)
        perturbation[j] = epsilon

        # Central difference
        f_plus = transition_func(state + perturbation)
        f_minus = transition_func(state - perturbation)

        jacobian[:, j] = (f_plus - f_minus) / (2 * epsilon)

    return jacobian


def analyze_eigenvalues(jacobian: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StabilityType]:
    """
    Analyze eigenvalues of Jacobian for stability classification.

    Per Section 3.1: A stationary state is locally stable if all
    eigenvalues lie inside the unit circle.

    Args:
        jacobian: The Jacobian matrix

    Returns:
        Tuple of (eigenvalues, eigenvectors, stability_type)
    """
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)

    # Compute moduli
    moduli = np.abs(eigenvalues)

    # Classification
    inside = moduli < 1.0 - 1e-10
    outside = moduli > 1.0 + 1e-10
    on_circle = ~inside & ~outside

    if np.all(inside):
        stability_type = StabilityType.STABLE
    elif np.any(on_circle):
        stability_type = StabilityType.MARGINAL  # Bifurcation point
    elif np.any(inside) and np.any(outside):
        stability_type = StabilityType.SADDLE
    else:
        stability_type = StabilityType.UNSTABLE

    return eigenvalues, eigenvectors, stability_type


class JacobianAnalyzer:
    """
    Analyzer for Jacobian stability following JEDC paper Section 3.

    Implements the block structure analysis from Eq. 6-7:

    J = [J_Y    J_T    J_M  ]
        [0      J_TT   J_TM ]
        [0      J_MT   J_MM ]

    Where:
    - J_Y: Output dynamics block
    - J_T: Tension impact on output
    - J_M: Memory impact on output
    - J_TT: Tension self-dynamics
    - J_TM: Tension-memory cross effects
    - J_MT: Memory-tension cross effects
    - J_MM ≈ (1-δ): Memory persistence block
    """

    def __init__(
        self,
        state_dim: int = 6,
        output_dim: int = 4,  # Y, g, a, θ
        tension_dim: int = 5,  # T_E, T_C, T_D, T_F, T_X
        memory_dim: int = 3,  # micro, meso, macro
        memory_decay: float = 0.1
    ):
        """
        Initialize the Jacobian analyzer.

        Args:
            state_dim: Total dimension of aggregate state S_t
            output_dim: Dimension of output variables (Y, g, a, θ)
            tension_dim: Number of structural tensions
            memory_dim: Memory hierarchy levels
            memory_decay: δ parameter for memory decay (J_MM ≈ 1-δ)
        """
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.tension_dim = tension_dim
        self.memory_dim = memory_dim
        self.memory_decay = memory_decay

        # Block indices
        self.y_idx = slice(0, output_dim)
        self.t_idx = slice(output_dim, output_dim + tension_dim)
        self.m_idx = slice(output_dim + tension_dim,
                          output_dim + tension_dim + memory_dim)

    def construct_theoretical_jacobian(
        self,
        J_Y: np.ndarray,
        J_T: np.ndarray,
        J_M: np.ndarray,
        J_TT: np.ndarray,
        J_TM: np.ndarray,
        J_MT: np.ndarray
    ) -> np.ndarray:
        """
        Construct the full Jacobian from block components.

        Per Eq. 6-7, the Jacobian has block-triangular structure
        with J_MM ≈ (1-δ)I capturing memory persistence.
        """
        n_y = self.output_dim
        n_t = self.tension_dim
        n_m = self.memory_dim
        n_total = n_y + n_t + n_m

        J = np.zeros((n_total, n_total))

        # Output block (top row)
        J[:n_y, :n_y] = J_Y
        J[:n_y, n_y:n_y+n_t] = J_T
        J[:n_y, n_y+n_t:] = J_M

        # Tension block (middle row) - zeros in first column
        J[n_y:n_y+n_t, n_y:n_y+n_t] = J_TT
        J[n_y:n_y+n_t, n_y+n_t:] = J_TM

        # Memory block (bottom row) - zeros in first column
        J[n_y+n_t:, n_y:n_y+n_t] = J_MT
        J[n_y+n_t:, n_y+n_t:] = (1 - self.memory_decay) * np.eye(n_m)

        return J

    def extract_block_structure(self, jacobian: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract block structure from full Jacobian.

        Returns:
            Dictionary with block matrices: J_Y, J_T, J_M, J_TT, J_TM, J_MT, J_MM
        """
        n_y = self.output_dim
        n_t = self.tension_dim

        return {
            'J_Y': jacobian[:n_y, :n_y],
            'J_T': jacobian[:n_y, n_y:n_y+n_t],
            'J_M': jacobian[:n_y, n_y+n_t:],
            'J_TT': jacobian[n_y:n_y+n_t, n_y:n_y+n_t],
            'J_TM': jacobian[n_y:n_y+n_t, n_y+n_t:],
            'J_MT': jacobian[n_y+n_t:, n_y:n_y+n_t],
            'J_MM': jacobian[n_y+n_t:, n_y+n_t:]
        }

    def analyze_stability(
        self,
        transition_func: Callable,
        state: np.ndarray,
        epsilon: float = 1e-6
    ) -> StabilityResult:
        """
        Perform complete stability analysis at a state.

        Args:
            transition_func: State transition function F(S)
            state: State vector to analyze
            epsilon: Perturbation for numerical derivatives

        Returns:
            StabilityResult with full analysis
        """
        # Compute numerical Jacobian
        jacobian = compute_numerical_jacobian(transition_func, state, epsilon)

        # Eigenvalue analysis
        eigenvalues, eigenvectors, stability_type = analyze_eigenvalues(jacobian)

        # Spectral radius and dominant eigenvalue
        moduli = np.abs(eigenvalues)
        spectral_radius = np.max(moduli)
        dominant_idx = np.argmax(moduli)
        dominant_eigenvalue = eigenvalues[dominant_idx]

        # Convergence rate (if stable)
        convergence_rate = None
        if stability_type == StabilityType.STABLE:
            convergence_rate = -np.log(spectral_radius)  # Per-period rate

        # Oscillation period (if complex eigenvalues)
        oscillation_period = None
        if np.abs(dominant_eigenvalue.imag) > 1e-10:
            # Period = 2π / angle
            angle = np.angle(dominant_eigenvalue)
            if np.abs(angle) > 1e-10:
                oscillation_period = 2 * np.pi / np.abs(angle)

        # Extract block structure
        block_structure = None
        if jacobian.shape[0] == self.output_dim + self.tension_dim + self.memory_dim:
            block_structure = self.extract_block_structure(jacobian)

        return StabilityResult(
            jacobian=jacobian,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            stability_type=stability_type,
            spectral_radius=spectral_radius,
            dominant_eigenvalue=dominant_eigenvalue,
            convergence_rate=convergence_rate,
            oscillation_period=oscillation_period,
            block_structure=block_structure
        )

    def compute_memory_induced_persistence(self, jacobian: np.ndarray) -> float:
        """
        Compute persistence induced by memory block.

        Per Section 3.1: J_MM ≈ (1-δ) may slow convergence or
        generate quasi-cyclical behavior.

        Returns:
            Estimated half-life of memory effects
        """
        blocks = self.extract_block_structure(jacobian)
        J_MM = blocks['J_MM']

        # Eigenvalue of memory block
        memory_eigenvalues = np.linalg.eigvals(J_MM)
        max_memory_eigenvalue = np.max(np.abs(memory_eigenvalues))

        if max_memory_eigenvalue >= 1.0:
            return np.inf  # Non-decaying memory

        # Half-life: t such that λ^t = 0.5
        half_life = np.log(0.5) / np.log(max_memory_eigenvalue)
        return half_life


class BifurcationAnalyzer:
    """
    Analyzer for bifurcation detection and characterization.

    Identifies critical parameter values where eigenvalues cross
    the unit circle, indicating qualitative changes in dynamics.

    Per Section 3.3: Transitions correspond to smooth bifurcation-like
    changes where small perturbations can trigger large qualitative changes.
    """

    def __init__(
        self,
        jacobian_analyzer: JacobianAnalyzer,
        transition_func_factory: Callable[[float], Callable]
    ):
        """
        Initialize bifurcation analyzer.

        Args:
            jacobian_analyzer: JacobianAnalyzer instance
            transition_func_factory: Function that takes parameter value
                                    and returns transition function F(S)
        """
        self.jacobian_analyzer = jacobian_analyzer
        self.transition_func_factory = transition_func_factory

    def find_bifurcation_points(
        self,
        state: np.ndarray,
        param_range: Tuple[float, float],
        n_points: int = 100,
        tolerance: float = 0.05
    ) -> List[Dict]:
        """
        Find bifurcation points by scanning parameter range.

        Detects where spectral radius crosses 1.0 (unit circle).

        Args:
            state: Reference state for analysis
            param_range: (min, max) parameter values to scan
            n_points: Number of points to sample
            tolerance: Tolerance for detecting crossing

        Returns:
            List of bifurcation points with parameter values and types
        """
        param_values = np.linspace(param_range[0], param_range[1], n_points)
        spectral_radii = []
        stability_types = []

        for param in param_values:
            transition_func = self.transition_func_factory(param)
            result = self.jacobian_analyzer.analyze_stability(
                transition_func, state
            )
            spectral_radii.append(result.spectral_radius)
            stability_types.append(result.stability_type)

        spectral_radii = np.array(spectral_radii)

        # Find crossings of unit circle
        bifurcation_points = []
        for i in range(len(param_values) - 1):
            # Check for crossing
            r1, r2 = spectral_radii[i], spectral_radii[i+1]

            if (r1 < 1.0 - tolerance and r2 > 1.0 + tolerance) or \
               (r1 > 1.0 + tolerance and r2 < 1.0 - tolerance):
                # Linear interpolation to find crossing
                crossing_param = param_values[i] + (1.0 - r1) / (r2 - r1) * \
                                (param_values[i+1] - param_values[i])

                bifurcation_type = "fold" if r1 < r2 else "stabilization"

                bifurcation_points.append({
                    'parameter': crossing_param,
                    'type': bifurcation_type,
                    'from_stability': stability_types[i],
                    'to_stability': stability_types[i+1],
                    'spectral_radius_change': (r1, r2)
                })

        return bifurcation_points

    def compute_bifurcation_diagram(
        self,
        state: np.ndarray,
        param1_range: Tuple[float, float],
        param2_range: Tuple[float, float],
        n_points1: int = 50,
        n_points2: int = 50,
        param_factory: Callable[[float, float], Callable] = None
    ) -> Dict:
        """
        Compute 2D bifurcation diagram.

        Creates a grid showing stability regions as function of
        two parameters (e.g., tension and memory decay).

        Args:
            state: Reference state
            param1_range: Range for first parameter
            param2_range: Range for second parameter
            n_points1: Grid points for parameter 1
            n_points2: Grid points for parameter 2
            param_factory: Function(p1, p2) -> transition_func

        Returns:
            Dictionary with grid data and stability classification
        """
        if param_factory is None:
            param_factory = lambda p1, p2: self.transition_func_factory(p1)

        p1_values = np.linspace(param1_range[0], param1_range[1], n_points1)
        p2_values = np.linspace(param2_range[0], param2_range[1], n_points2)

        spectral_radii = np.zeros((n_points1, n_points2))
        stability_map = np.zeros((n_points1, n_points2), dtype=int)

        stability_codes = {
            StabilityType.STABLE: 0,
            StabilityType.MARGINAL: 1,
            StabilityType.SADDLE: 2,
            StabilityType.UNSTABLE: 3
        }

        for i, p1 in enumerate(p1_values):
            for j, p2 in enumerate(p2_values):
                transition_func = param_factory(p1, p2)
                result = self.jacobian_analyzer.analyze_stability(
                    transition_func, state
                )
                spectral_radii[i, j] = result.spectral_radius
                stability_map[i, j] = stability_codes[result.stability_type]

        return {
            'param1_values': p1_values,
            'param2_values': p2_values,
            'spectral_radii': spectral_radii,
            'stability_map': stability_map,
            'stability_codes': stability_codes
        }

    def analyze_cusp_catastrophe(
        self,
        state: np.ndarray,
        tension_range: Tuple[float, float],
        memory_range: Tuple[float, float],
        n_points: int = 50
    ) -> Dict:
        """
        Analyze cusp catastrophe structure per Section 3.3.

        The cusp is characterized by:
        - Two stable equilibria at low tension
        - Single stable equilibrium at high tension
        - Catastrophic transitions at fold lines

        Returns:
            Dictionary with cusp surface data and fold curves
        """
        # This requires solving for equilibria at each parameter value
        # Simplified version: compute stability boundaries

        tensions = np.linspace(tension_range[0], tension_range[1], n_points)
        memories = np.linspace(memory_range[0], memory_range[1], n_points)

        T, M = np.meshgrid(tensions, memories)

        # Theoretical cusp: positions where system has multiple equilibria
        # Based on cusp normal form: x^3 + ax + b = 0
        # Bifurcation set: 4a^3 + 27b^2 = 0

        # Map (tension, memory) to cusp parameters
        a = -3 * (M - 0.5)**2 / 2
        b = (T - 0.5)**3

        # Compute bifurcation discriminant
        discriminant = 4 * a**3 + 27 * b**2

        # Regions
        multiple_equilibria = discriminant < 0

        # Fold curves (cusp boundary)
        fold_curve_t = tensions
        fold_curve_m_upper = 0.5 + np.sqrt(2/3) * np.abs(tensions - 0.5)
        fold_curve_m_lower = 0.5 - np.sqrt(2/3) * np.abs(tensions - 0.5)

        return {
            'tension_grid': T,
            'memory_grid': M,
            'discriminant': discriminant,
            'multiple_equilibria_region': multiple_equilibria,
            'fold_curve': {
                'tension': fold_curve_t,
                'memory_upper': fold_curve_m_upper,
                'memory_lower': fold_curve_m_lower
            }
        }


def compute_amplification_factor(
    jacobian: np.ndarray,
    degree_distribution: np.ndarray
) -> float:
    """
    Compute amplification factor from network structure.

    Per Eq. 16: E[Δg_t²] ∝ Σ_i k_i² E[Δx_{i,t}²]

    For scale-free networks with 2 < γ < 3, the second moment
    diverges, implying structural amplification.

    Args:
        jacobian: System Jacobian
        degree_distribution: Array of node degrees k_i

    Returns:
        Amplification factor A = Σ k_i² / N
    """
    k_squared = degree_distribution ** 2
    amplification = np.mean(k_squared)

    # Compare to random network baseline (E[k²] = E[k]² + Var[k])
    mean_k = np.mean(degree_distribution)
    baseline = mean_k ** 2

    # Amplification ratio
    amplification_ratio = amplification / baseline if baseline > 0 else np.inf

    return amplification_ratio


def verify_second_moment_divergence(
    degree_distribution: np.ndarray,
    gamma_estimate: float
) -> Dict:
    """
    Verify second moment divergence for scale-free network.

    For power law P(k) ~ k^(-γ):
    - γ > 3: E[k²] finite
    - 2 < γ < 3: E[k²] diverges as N → ∞

    Returns:
        Analysis of moment convergence
    """
    k_squared = degree_distribution ** 2

    # Compute running variance to check for divergence
    n = len(degree_distribution)
    running_mean = np.cumsum(k_squared) / np.arange(1, n + 1)

    # Check if variance is increasing with sample size
    variance_trend = np.polyfit(np.log(np.arange(10, n)),
                                np.log(running_mean[9:]), 1)

    is_diverging = variance_trend[0] > 0.1  # Positive slope indicates divergence

    return {
        'gamma_estimate': gamma_estimate,
        'second_moment_finite': gamma_estimate > 3.0,
        'empirical_divergence': is_diverging,
        'variance_trend_slope': variance_trend[0],
        'mean_k_squared': np.mean(k_squared),
        'max_k_squared': np.max(k_squared)
    }
