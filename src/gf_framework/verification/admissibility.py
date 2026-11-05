from __future__ import annotations

"""
GF admissibility checks for Sanchez-Moreno-style formal verification claims.

This module validates representational and structural assumptions that are
required by the current codebase's formal GF verification paths. It does not
prove Lyapunov-function existence.

Status: TEST-ONLY (reclassified by WP P6-06, 2026-09-02). Outside its
re-export in ``verification/__init__.py`` the only importer is
``tests/test_verification.py``. Nothing on the certificate path
(``certify_discovered_gf`` → ``joint_sos`` → ``_validate_gram_certificate``)
calls it, so a passing admissibility report is never evidence for a claim.
"""

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gf_framework.algebra.core import CompositeGFTerm, GeneralizedForm, GFTerm


@dataclass(frozen=True)
class AdmissibilityCheck:
    name: str
    status: str  # "pass" | "warn" | "fail"
    message: str


@dataclass
class GFSystemAdmissibilityReport:
    system_name: str
    in_scope_formal_gf: bool
    severity: str  # "pass" | "warn" | "fail"
    checks: List[AdmissibilityCheck] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [asdict(c) for c in self.checks]
        return payload


def _status_rank(status: str) -> int:
    norm = str(status).strip().lower()
    if norm == "fail":
        return 2
    if norm == "warn":
        return 1
    return 0


def _merge_severity(checks: Sequence[AdmissibilityCheck]) -> str:
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warn" for c in checks):
        return "warn"
    return "pass"


def _iter_all_terms_recursive(gf: GeneralizedForm) -> Iterable[GFTerm]:
    for term in getattr(gf, "terms", []) or []:
        yield term
    for ct in getattr(gf, "composite_terms", []) or []:
        if isinstance(ct, CompositeGFTerm):
            yield from _iter_all_terms_recursive(ct.inner_gf)


def _iter_all_composites_recursive(gf: GeneralizedForm) -> Iterable[CompositeGFTerm]:
    for ct in getattr(gf, "composite_terms", []) or []:
        if isinstance(ct, CompositeGFTerm):
            yield ct
            yield from _iter_all_composites_recursive(ct.inner_gf)


def _gf_component_degree_set(gf: GeneralizedForm, weights_r: Tuple[int, ...]) -> set[Fraction]:
    degs: set[Fraction] = set()
    for term in getattr(gf, "terms", []) or []:
        degs.add(term.weighted_degree(weights_r))
    for ct in getattr(gf, "composite_terms", []) or []:
        degs.add(ct.weighted_degree(weights_r))
    return degs


def _is_fraction_like(x: Any) -> bool:
    return isinstance(x, Fraction)


def _validate_gf_component(
    gf: Any,
    *,
    component_index: int,
    weights_r: Tuple[int, ...],
    expected_degree: Optional[Fraction],
) -> List[AdmissibilityCheck]:
    checks: List[AdmissibilityCheck] = []
    label = f"f[{component_index}]"
    if not isinstance(gf, GeneralizedForm):
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_type",
                status="fail",
                message=f"Component is not GeneralizedForm (got {type(gf).__name__})",
            )
        )
        return checks

    if tuple(getattr(gf, "weights_r", ())) != tuple(weights_r):
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_weights_match",
                status="fail",
                message=f"GF weights {getattr(gf, 'weights_r', None)} != system weights {weights_r}",
            )
        )
    else:
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_weights_match",
                status="pass",
                message="GF weights match system weights",
            )
        )

    for term in _iter_all_terms_recursive(gf):
        if any(not _is_fraction_like(exp) for exp in getattr(term, "exponents", ())):
            checks.append(
                AdmissibilityCheck(
                    name=f"{label}_rational_exponents",
                    status="fail",
                    message="Found non-rational standard-term exponent(s)",
                )
            )
            break
    else:
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_rational_exponents",
                status="pass",
                message="All standard-term exponents are Fractions",
            )
        )

    composites = list(_iter_all_composites_recursive(gf))
    if composites:
        bad_outer = any(not _is_fraction_like(ct.outer_exponent) for ct in composites)
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_composite_terms",
                status="fail",
                message=(
                    "Composite/nested GF terms present; current formal GF admissibility "
                    "validator marks this out-of-scope"
                ),
            )
        )
        if bad_outer:
            checks.append(
                AdmissibilityCheck(
                    name=f"{label}_composite_outer_exponents",
                    status="fail",
                    message="Found non-rational composite outer exponent(s)",
                )
            )
    else:
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_composite_terms",
                status="pass",
                message="No composite/nested GF terms",
            )
        )

    degs = _gf_component_degree_set(gf, weights_r)
    if not degs:
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_nonempty_degree_set",
                status="fail",
                message="No terms found in dynamics component",
            )
        )
    elif len(degs) != 1:
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_homogeneous_degree",
                status="fail",
                message=f"Mixed weighted degrees detected: {[str(d) for d in sorted(degs)]}",
            )
        )
    else:
        actual = next(iter(degs))
        if expected_degree is not None and actual != expected_degree:
            checks.append(
                AdmissibilityCheck(
                    name=f"{label}_kappa_consistency",
                    status="fail",
                    message=f"Weighted degree {actual} != expected kappa+r_i degree {expected_degree}",
                )
            )
        else:
            checks.append(
                AdmissibilityCheck(
                    name=f"{label}_homogeneous_degree",
                    status="pass",
                    message=f"Single weighted degree {actual}",
                )
            )

    try:
        mu = gf.mu_exponents
        mu_ok = (
            isinstance(mu, tuple)
            and len(mu) == len(weights_r)
            and all(isinstance(v, int) and v > 0 for v in mu)
        )
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_mu_transform",
                status="pass" if mu_ok else "fail",
                message=f"mu_exponents={mu}" if mu_ok else f"Invalid mu_exponents={mu}",
            )
        )
    except Exception as exc:
        checks.append(
            AdmissibilityCheck(
                name=f"{label}_mu_transform",
                status="fail",
                message=f"mu_exponents computation failed: {exc}",
            )
        )

    return checks


def validate_dynamics_gf_admissibility(
    *,
    dynamics_gf: Sequence[Any],
    weights_r: Sequence[int],
    degree_m: Any,
    kappa: Optional[float],
    system_name: str = "unnamed",
) -> GFSystemAdmissibilityReport:
    checks: List[AdmissibilityCheck] = []
    weights_tuple = tuple(weights_r)

    weights_ok = (
        isinstance(weights_tuple, tuple)
        and len(weights_tuple) > 0
        and all(isinstance(w, int) and w > 0 for w in weights_tuple)
    )
    checks.append(
        AdmissibilityCheck(
            name="weights_r_positive_ints",
            status="pass" if weights_ok else "fail",
            message=f"weights_r={weights_tuple}",
        )
    )

    degree_ok = isinstance(degree_m, int)
    checks.append(
        AdmissibilityCheck(
            name="degree_m_integer",
            status="pass" if degree_ok else "fail",
            message=f"degree_m={degree_m!r}",
        )
    )

    dyn_list = list(dynamics_gf) if dynamics_gf is not None else []
    dyn_len_ok = len(dyn_list) == len(weights_tuple) and len(dyn_list) > 0
    checks.append(
        AdmissibilityCheck(
            name="dynamics_dim_match",
            status="pass" if dyn_len_ok else "fail",
            message=f"len(dynamics_gf)={len(dyn_list)}, len(weights_r)={len(weights_tuple)}",
        )
    )

    if kappa is None:
        kappa_fraction = None
        checks.append(
            AdmissibilityCheck(
                name="kappa_provided",
                status="warn",
                message="kappa not provided; component-degree consistency not checked",
            )
        )
    else:
        try:
            kappa_fraction = Fraction(str(kappa))
            checks.append(
                AdmissibilityCheck(
                    name="kappa_provided",
                    status="pass",
                    message=f"kappa={kappa_fraction}",
                )
            )
        except Exception as exc:
            kappa_fraction = None
            checks.append(
                AdmissibilityCheck(
                    name="kappa_provided",
                    status="fail",
                    message=f"invalid kappa={kappa!r}: {exc}",
                )
            )

    if weights_ok and dyn_len_ok:
        for i, gf in enumerate(dyn_list):
            expected_degree = None
            if kappa_fraction is not None:
                expected_degree = kappa_fraction + Fraction(weights_tuple[i])
            checks.extend(
                _validate_gf_component(
                    gf,
                    component_index=i,
                    weights_r=weights_tuple,
                    expected_degree=expected_degree,
                )
            )

    severity = _merge_severity(checks)
    in_scope = not any(
        c.status == "fail"
        and (
            c.name.endswith("_composite_terms")
            or c.name in {"weights_r_positive_ints", "degree_m_integer", "dynamics_dim_match"}
            or "homogeneous" in c.name
            or "rational_exponents" in c.name
            or "kappa_consistency" in c.name
            or "mu_transform" in c.name
            or c.name.endswith("_type")
        )
        for c in checks
    )

    assumptions = [
        "Validator checks representational prerequisites supported by this codebase (GF structure, homogeneity, rational exponents).",
        "Validator does not prove existence of a Lyapunov function in the GF class.",
    ]
    limitations = [
        "Composite/nested dynamics are marked out-of-scope for formal GF admissibility by current policy.",
        "kappa consistency is only checked when kappa is provided.",
    ]
    if in_scope:
        summary = "System appears in-scope for formal GF verification under current representational checks."
    else:
        summary = "System fails one or more formal GF admissibility checks used for rigorous in-scope claims."

    return GFSystemAdmissibilityReport(
        system_name=str(system_name),
        in_scope_formal_gf=bool(in_scope),
        severity=severity,
        checks=checks,
        assumptions=assumptions,
        limitations=limitations,
        summary=summary,
    )


def validate_system_spec_admissibility(spec: Any) -> GFSystemAdmissibilityReport:
    return validate_dynamics_gf_admissibility(
        dynamics_gf=getattr(spec, "dynamics_gf", None),
        weights_r=getattr(spec, "weights_r", ()),
        degree_m=getattr(spec, "degree_m", None),
        kappa=getattr(spec, "kappa", None),
        system_name=str(getattr(spec, "name", "unnamed")),
    )


def validate_matrix_case_admissibility(case: Any, *, kappa: Optional[float] = None) -> GFSystemAdmissibilityReport:
    case_kappa = getattr(case, "kappa", None)
    if kappa is None:
        kappa = case_kappa
    return validate_dynamics_gf_admissibility(
        dynamics_gf=getattr(case, "dynamics_gf", None),
        weights_r=getattr(case, "weights_r", ()),
        degree_m=getattr(case, "degree_m", None),
        kappa=kappa,
        system_name=str(getattr(case, "name", "unnamed")),
    )
