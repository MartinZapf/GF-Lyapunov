"""Benchmark controller systems and dynamics definitions."""

from __future__ import annotations

from gf_framework.systems.base import System
from gf_framework.systems.fosmc import FOSMCSystem
from gf_framework.systems.power_rate import PowerRateSystem
from gf_framework.systems.sta import STASystem
from gf_framework.systems.twisting import TwistingSystem
from gf_framework.systems.example1 import Example1System
from gf_framework.systems.example3 import Example3System
from gf_framework.systems.continuous_twisting import CTASystem
from gf_framework.systems.levant3d import Levant3DSystem
from gf_framework.systems.pidsmc import PIDSMCSystem

CONTROLLER_REGISTRY = {
    "fosmc": FOSMCSystem,
    "power_rate": PowerRateSystem,
    "sta": STASystem,
    "twisting": TwistingSystem,
    "example1": Example1System,
    "example3": Example3System,
    "cta": CTASystem,
    "levant3d": Levant3DSystem,
    "pidsmc": PIDSMCSystem,
}


def get_system(name: str, **kwargs):
    """Look up a system class by name and instantiate it."""
    cls = CONTROLLER_REGISTRY[name]
    return cls(**kwargs) if kwargs else cls()


__all__ = [
    "System",
    "FOSMCSystem",
    "PowerRateSystem",
    "STASystem",
    "TwistingSystem",
    "Example1System",
    "Example3System",
    "CTASystem",
    "Levant3DSystem",
    "PIDSMCSystem",
    "CONTROLLER_REGISTRY",
    "get_system",
]
