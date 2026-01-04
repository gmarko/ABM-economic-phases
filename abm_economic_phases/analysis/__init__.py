"""
Analysis module for stability, bifurcation, and comparative country analysis.

Implements:
- Jacobian computation, eigenvalue analysis, and bifurcation detection (Section 3)
- Country-specific calibration profiles (Section 6.3)
- Cross-country comparative analysis tools
"""

from .stability import (
    JacobianAnalyzer,
    StabilityResult,
    BifurcationAnalyzer,
    compute_numerical_jacobian,
    analyze_eigenvalues,
)

from .country_profiles import (
    CountryProfile,
    CountryComparator,
    COUNTRY_PROFILES,
    SPAIN,
    GERMANY,
    FRANCE,
    UK,
    USA,
    CHINA,
)

__all__ = [
    # Stability analysis
    'JacobianAnalyzer',
    'StabilityResult',
    'BifurcationAnalyzer',
    'compute_numerical_jacobian',
    'analyze_eigenvalues',
    # Country profiles
    'CountryProfile',
    'CountryComparator',
    'COUNTRY_PROFILES',
    'SPAIN',
    'GERMANY',
    'FRANCE',
    'UK',
    'USA',
    'CHINA',
]
