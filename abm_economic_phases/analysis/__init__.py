"""
Analysis module for stability and bifurcation analysis.

Implements Jacobian computation, eigenvalue analysis, and bifurcation detection
as specified in Section 3 of the JEDC paper.
"""

from .stability import (
    JacobianAnalyzer,
    StabilityResult,
    BifurcationAnalyzer,
    compute_numerical_jacobian,
    analyze_eigenvalues,
)

__all__ = [
    'JacobianAnalyzer',
    'StabilityResult',
    'BifurcationAnalyzer',
    'compute_numerical_jacobian',
    'analyze_eigenvalues',
]
