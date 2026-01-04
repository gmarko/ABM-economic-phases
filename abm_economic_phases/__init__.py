"""
ABM Economic Phases Model

A complex systems approach to economic phase dynamics with:
- Heterogeneous agents (Households, Firms, Banks, Government)
- Scale-free network topology
- Adaptive memory and reinforcement learning
- Black Swan and Unicorn events
- MMT compatibility
"""

from .simulation.engine import EconomicSimulation
from .economy.phases import EconomicPhase
from .utils.parameters import ModelParameters

__version__ = "0.1.0"
__author__ = "Marco Durán Cabobianco"

__all__ = [
    "EconomicSimulation",
    "EconomicPhase",
    "ModelParameters",
]
