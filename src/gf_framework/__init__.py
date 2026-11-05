"""
GF Framework — Automatic discovery and formal certification of Lyapunov functions.

Based on the Generalized Forms (GF) theory of Sanchez & Moreno (2018/2019),
this framework automates Lyapunov function discovery for homogeneous systems
via SOS design over the μ-compatible C¹ GF library and certifies results
through a strict SOS gate with a-posteriori Gram validation.

Public API:
    gf_framework.algebra     — GF types, term enumeration, associated forms
    gf_framework.systems     — Benchmark controllers and dynamics
    gf_framework.verification — Formal SOS/Pólya certification
    gf_framework.discovery   — GF-SOS discovery pipeline
    gf_framework.interfaces  — Shared dataclass interfaces
"""

__version__ = "1.0.0"
