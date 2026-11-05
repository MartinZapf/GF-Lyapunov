"""
Parity and symmetry reduction of the GF-SOS problem (campaign item C1, block S1).

Solver-free by design: every assertion here is symbolic (sympy) or purely
combinatorial. No SDP is built and no solver is called, so this file runs in CI
on machines without MOSEK.

Covered:

* the benchmark dynamics are odd, checked symbolically per system;
* a deliberately non-odd toy field is rejected, with a nonzero residual;
* the even-parity filter keeps exactly the even terms of a hand-checked STA
  library, and only even terms on every benchmark library;
* an even V has identical associated forms on antipodal octants (the fact the
  nominal halving rests on);
* the nominal pairing has 2^(n-1) representatives and is a fixed-point-free
  involution for n = 1..4;
* the perturbed pairing SWAPS the endpoints, and nominal halving cannot be
  requested once endpoints are present;
* the paired identity W_+(-x) = W_-(x) holds while the naive one
  W_+(-x) = W_+(x) does not.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy as sp

import gf_framework.discovery.universal_discovery as ud
from gf_framework.algebra.associated import (
    get_hyperoctant_signs,
    gf_to_associated_forms,
)
from gf_framework.algebra.core import GeneralizedForm, GFTerm
from gf_framework.algebra.enumerate import TermSpec, enumerate_diagonal_terms
from gf_framework.algebra.lie_derivative import compute_lie_derivative_gf
from gf_framework.algebra.symmetry import (
    ENDPOINT_MINUS,
    ENDPOINT_PLUS,
    PARITY_EVEN,
    PARITY_ODD,
    NominalParityReduction,
    PairedParityReduction,
    ParityReductionError,
    build_parity_reduction,
    endpoint_count,
    filter_even_parity_terms,
    flip_endpoint,
    flip_octant,
    gf_parity,
    is_even_term,
    nominal_octant_reduction,
    octant_from_index,
    octant_index,
    paired_endpoint_reduction,
    plan_parity_reduction,
    split_library_by_parity,
    term_parity,
    validate_library_split,
    validate_reduction,
    verify_dynamics_odd,
    verify_endpoint_swap,
)
from gf_framework.discovery.gf_discovery import mu_filtered_cross
from gf_framework.systems.sta import get_sta_known_lyapunov

# The 12 systems of the campaign matrix (11 benchmark + levant4d).
BENCHMARK_FACTORIES = {
    "fosmc": "create_fosmc_spec",
    "power_rate": "create_power_rate_spec",
    "sta": "create_sta_spec",
    "twisting": "create_twisting_spec",
    "example1": "create_example1_spec",
    "example3": "create_example3_spec",
    "levant3d": "create_levant3d_spec",
    "pidsmc": "create_pidsmc_spec",
    "dsta": "create_dsta_spec",
    "fc_cta": "create_fc_cta_spec",
    "continuous_twisting": "create_cta_spec",
    "levant4d": "create_levant4d_spec",
}

# Systems that admit a constant matched disturbance (weighted-degree-0 channel).
# power_rate, example1 and example3 are structurally excluded, see the
# disturbance-model section of the campaign file.
PERTURBED_FACTORIES = {
    "fosmc": ("create_fosmc_spec", 1.0),
    "sta": ("create_sta_spec", 1.0),
    "twisting": ("create_twisting_spec", 0.5),
    "levant3d": ("create_levant3d_spec", 1.0),
    "pidsmc": ("create_pidsmc_spec", 0.5),
    "dsta": ("create_dsta_spec", 0.3),
    "fc_cta": ("create_fc_cta_spec", 0.5),
    "continuous_twisting": ("create_cta_spec", 0.5),
    "levant4d": ("create_levant4d_spec", 0.3),
}


def _spec(name: str):
    return getattr(ud, BENCHMARK_FACTORIES[name])()


def _perturbed_spec(name: str):
    factory, delta0 = PERTURBED_FACTORIES[name]
    return getattr(ud, factory)(delta0=delta0)


def _diagonal_gf(degree_m, weights_r) -> GeneralizedForm:
    """V = sum_i |x_i|^(m/r_i): even, positive definite, no cross terms."""
    n = len(weights_r)
    terms = [
        GFTerm(
            coefficient=1.0,
            exponents=tuple(
                Fraction(degree_m, weights_r[i]) if j == i else Fraction(0)
                for j in range(n)
            ),
            term_types=("abs",) * n,
        )
        for i in range(n)
    ]
    return GeneralizedForm(degree_m=degree_m, weights_r=weights_r, terms=terms)


def _to_sympy(gf, symbols):
    expr = sp.Integer(0)
    for t in gf.terms:
        expr = expr + t.to_sympy(symbols)
    return expr


# =============================================================================
# Term parity
# =============================================================================

def test_term_parity_counts_signed_axes():
    even = TermSpec((Fraction(1, 2), Fraction(1)), ("signed", "signed"))
    odd = TermSpec((Fraction(1, 2), Fraction(1)), ("signed", "abs"))
    plain = TermSpec((Fraction(1), Fraction(0)), ("abs", "abs"))

    assert term_parity(even) == PARITY_EVEN
    assert term_parity(odd) == PARITY_ODD
    assert term_parity(plain) == PARITY_EVEN
    assert is_even_term(even) and not is_even_term(odd)


def test_signed_exponent_zero_still_flips():
    """sign(x) enters at exponent 0 too, so such a term is odd."""
    relay = TermSpec((Fraction(0), Fraction(0)), ("signed", "abs"))
    assert term_parity(relay) == PARITY_ODD


# =============================================================================
# Symbolic oddness of the benchmark dynamics
# =============================================================================

@pytest.mark.parametrize("name", sorted(BENCHMARK_FACTORIES))
def test_benchmark_dynamics_are_odd(name):
    spec = _spec(name)
    check = verify_dynamics_odd(spec.dynamics_gf)
    assert check.holds, f"{name}: {check.message} residuals={check.residuals}"
    assert check.identity == "f(-x) = -f(x)"
    assert check.offending == ()
    assert all(r == "0" for r in check.residuals)
    # Structural parity must agree with the symbolic verdict.
    assert all(p == PARITY_ODD for p in check.structural_parity)
    assert all(gf_parity(f) == PARITY_ODD for f in spec.dynamics_gf)


def test_non_odd_toy_dynamics_is_rejected():
    """f(x) = |x| is even, so the oddness check must reject it with a residual."""
    even_field = GeneralizedForm(
        degree_m=1,
        weights_r=(1,),
        terms=[GFTerm(1.0, (Fraction(1),), ("abs",))],
    )
    check = verify_dynamics_odd([even_field])

    assert not check.holds
    assert bool(check) is False
    assert check.offending == (0,)
    residual = sp.simplify(check.residual_exprs[0])
    assert residual != 0
    # The residual is f(-x) + f(x) = 2|x|, the object a failure must hand back.
    x1 = sp.Symbol("x1", real=True)
    assert float(residual.subs(x1, -3.0)) == pytest.approx(6.0)
    assert float(residual.subs(x1, 2.0)) == pytest.approx(4.0)


def test_partly_odd_toy_dynamics_is_rejected():
    """One even term inside an otherwise odd component is enough to fail."""
    x1, x2 = sp.symbols("x1 x2", real=True)
    mixed = GeneralizedForm(
        degree_m=1,
        weights_r=(2, 1),
        terms=[
            GFTerm(1.0, (Fraction(0), Fraction(1)), ("abs", "signed")),  # x2, odd
            GFTerm(1.0, (Fraction(1, 2), Fraction(0)), ("abs", "abs")),  # |x1|^(1/2)
        ],
    )
    odd_component = GeneralizedForm(
        degree_m=0,
        weights_r=(2, 1),
        terms=[GFTerm(-1.0, (Fraction(0), Fraction(0)), ("signed", "abs"))],
    )
    check = verify_dynamics_odd([mixed, odd_component])

    assert not check.holds
    assert check.offending == (0,)
    assert check.residuals[1] == "0"
    assert sp.simplify(check.residual_exprs[0] - 2 * sp.Abs(x1) ** 0.5) == 0
    assert gf_parity(mixed) is None  # mixed parity, no definite one


# =============================================================================
# Even-parity sub-library
# =============================================================================

def test_even_filter_on_hand_checked_sta_library():
    """STA (r = (2,1)) at m = 4: the mu-compatible library has 6 terms.

    Hand check, parity = (-1)^(number of signed axes):
        |x1|^2                    abs,abs        even   keep
        |x2|^4                    abs,abs        even   keep
        ceil(x1)^(3/2) ceil(x2)   signed,signed  even   keep
        ceil(x1) ceil(x2)^2       signed,signed  even   keep
        ceil(x1) |x2|^2           signed,abs     odd    drop
        |x1|^(3/2) ceil(x2)       abs,signed     odd    drop
    """
    library = [
        TermSpec((Fraction(2), Fraction(0)), ("abs", "abs")),
        TermSpec((Fraction(0), Fraction(4)), ("abs", "abs")),
        TermSpec((Fraction(3, 2), Fraction(1)), ("signed", "signed")),
        TermSpec((Fraction(1), Fraction(2)), ("signed", "signed")),
        TermSpec((Fraction(1), Fraction(2)), ("signed", "abs")),
        TermSpec((Fraction(3, 2), Fraction(1)), ("abs", "signed")),
    ]
    even, odd = split_library_by_parity(library, weights_r=(2, 1))

    assert even == library[:4]
    assert odd == library[4:]
    assert filter_even_parity_terms(library, weights_r=(2, 1)) == library[:4]
    assert validate_library_split(library, even, odd).ok


def test_even_filter_matches_the_enumerated_sta_library():
    """The hand-checked set is the library the pipeline actually enumerates."""
    spec = _spec("sta")
    diag = enumerate_diagonal_terms(4, spec.weights_r)
    _, _, cross = mu_filtered_cross(4, spec.weights_r, diag, spec.dynamics_gf)
    library = diag + cross

    even, odd = split_library_by_parity(library, weights_r=spec.weights_r)
    assert len(library) == 6
    assert len(even) == 4
    assert {(t.exponents, t.term_types) for t in odd} == {
        ((Fraction(1), Fraction(2)), ("signed", "abs")),
        ((Fraction(3, 2), Fraction(1)), ("abs", "signed")),
    }


@pytest.mark.parametrize("name", sorted(BENCHMARK_FACTORIES))
def test_even_filter_invariants_on_benchmark_libraries(name):
    spec = _spec(name)
    diag = enumerate_diagonal_terms(spec.degree_m, spec.weights_r)
    _, _, cross = mu_filtered_cross(
        spec.degree_m, spec.weights_r, diag, spec.dynamics_gf
    )
    library = diag + cross

    even, odd = split_library_by_parity(library, weights_r=spec.weights_r)
    report = validate_library_split(library, even, odd)

    assert report.ok, report.failures()
    assert all(is_even_term(t) for t in even)
    assert all(not is_even_term(t) for t in odd)
    assert len(even) + len(odd) == len(library)
    # The diagonal backbone is pure |x_i|^p and always survives.
    assert all(t in even for t in diag)


def test_associated_forms_agree_on_antipodal_octants_for_even_v():
    """The identity the nominal halving rests on, checked on the STA V."""
    V = get_sta_known_lyapunov()
    assert gf_parity(V) == PARITY_EVEN

    forms = gf_to_associated_forms(V)
    for signs in get_hyperoctant_signs(V.n_vars):
        assert sp.simplify(forms[signs] - forms[flip_octant(signs)]) == 0


def test_associated_forms_differ_when_an_odd_term_is_present():
    """Guard: without the even filter the octant targets are genuinely different."""
    V = get_sta_known_lyapunov()
    V.terms.append(GFTerm(0.5, (Fraction(1), Fraction(0)), ("signed", "abs")))
    assert gf_parity(V) is None

    forms = gf_to_associated_forms(V)
    differing = [
        signs
        for signs in get_hyperoctant_signs(V.n_vars)
        if sp.simplify(forms[signs] - forms[flip_octant(signs)]) != 0
    ]
    assert differing, "an odd term must break the antipodal identity"


# =============================================================================
# Octant bookkeeping
# =============================================================================

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_octant_index_matches_the_framework_enumeration(n):
    octants = get_hyperoctant_signs(n)
    for idx, signs in enumerate(octants):
        assert octant_index(signs) == idx
        assert octant_from_index(idx, n) == signs
    assert flip_octant(flip_octant(octants[0])) == octants[0]


def test_octant_index_rejects_non_sign_entries():
    with pytest.raises(ParityReductionError):
        octant_index((1, 0))


# =============================================================================
# Nominal reduction
# =============================================================================

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_nominal_reduction_halves_the_octants(n):
    red = nominal_octant_reduction(n, None)

    assert isinstance(red, NominalParityReduction)
    assert red.n_octants == 2 ** n
    assert len(red.representatives) == 2 ** (n - 1)
    assert red.n_v_blocks == 2 ** (n - 1)
    assert red.n_w_blocks == 2 ** (n - 1)
    assert not red.has_endpoints


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_nominal_reduction_is_a_fixed_point_free_involution(n):
    red = nominal_octant_reduction(n, None)
    octants = set(get_hyperoctant_signs(n))

    for signs in octants:
        assert red.partner(signs) == flip_octant(signs)
        assert red.partner(red.partner(signs)) == signs
        assert red.partner(signs) != signs

    reps = set(red.representatives)
    imgs = set(red.images)
    assert reps | imgs == octants
    assert not (reps & imgs)
    assert all(red.is_representative(s) for s in red.representatives)
    assert not any(red.is_representative(s) for s in red.images)

    report = validate_reduction(red)
    assert report.ok, report.failures()


def test_nominal_reduction_rejects_endpoint_labels():
    red = nominal_octant_reduction(2, None)
    assert red.w_blocks() == [(s, None) for s in red.representatives]
    with pytest.raises(ParityReductionError):
        red.partner_label(((1, 1), ENDPOINT_PLUS))


# =============================================================================
# Paired (perturbed) reduction
# =============================================================================

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_paired_reduction_sizes(n):
    red = paired_endpoint_reduction(n, 2)

    assert isinstance(red, PairedParityReduction)
    assert red.has_endpoints
    assert red.n_endpoints == 2
    # V never sees the disturbance and still halves.
    assert red.n_v_blocks == 2 ** (n - 1)
    # W: 2^(n+1) blocks paired down to 2^n, NOT to 2^(n-1).
    assert red.n_w_blocks == 2 ** n
    assert len(red.full_labels()) == 2 ** (n + 1)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_paired_reduction_swaps_the_endpoints(n):
    red = paired_endpoint_reduction(n, 2)

    for signs in get_hyperoctant_signs(n):
        # The explicit pairing: (s, +) is certified together with (-s, -).
        assert red.partner(signs, ENDPOINT_PLUS) == (flip_octant(signs), ENDPOINT_MINUS)
        assert red.partner(signs, ENDPOINT_MINUS) == (flip_octant(signs), ENDPOINT_PLUS)
        # Naive halving would claim (-s, +), which the map never returns.
        assert red.partner(signs, ENDPOINT_PLUS) != (flip_octant(signs), ENDPOINT_PLUS)
        for endpoint in (ENDPOINT_PLUS, ENDPOINT_MINUS):
            label = (signs, endpoint)
            assert red.partner_label(red.partner_label(label)) == label
            assert red.partner_label(label) != label

    reps = set(red.representatives)
    imgs = set(red.images)
    assert reps | imgs == set(red.full_labels())
    assert not (reps & imgs)
    assert len(reps) == 2 ** n
    # Both endpoints of every representative octant are kept.
    assert {e for _, e in reps} == {ENDPOINT_PLUS, ENDPOINT_MINUS}

    report = validate_reduction(red)
    assert report.ok, report.failures()


def test_flip_endpoint_is_an_involution():
    assert flip_endpoint(ENDPOINT_PLUS) == ENDPOINT_MINUS
    assert flip_endpoint(ENDPOINT_MINUS) == ENDPOINT_PLUS
    assert flip_endpoint(flip_endpoint(ENDPOINT_PLUS)) == ENDPOINT_PLUS
    assert flip_endpoint(None) is None
    with pytest.raises(ParityReductionError):
        flip_endpoint(7)


# =============================================================================
# The API must refuse nominal halving once endpoints are present
# =============================================================================

def test_nominal_halving_refused_when_endpoints_are_present():
    spec = _perturbed_spec("sta")
    vertices = spec.dynamics_gf_vertices
    assert len(vertices) == 2

    with pytest.raises(ParityReductionError, match="unsound"):
        nominal_octant_reduction(spec.n_vars, vertices)
    with pytest.raises(ParityReductionError, match="unsound"):
        nominal_octant_reduction(spec.n_vars, 2)

    # The dispatcher routes the same input to the paired reduction.
    red = build_parity_reduction(spec.n_vars, vertices)
    assert isinstance(red, PairedParityReduction)


def test_paired_reduction_refused_without_endpoints():
    with pytest.raises(ParityReductionError):
        paired_endpoint_reduction(2, None)
    with pytest.raises(ParityReductionError):
        paired_endpoint_reduction(2, 1)


def test_vertex_argument_is_mandatory_and_validated():
    with pytest.raises(TypeError):
        build_parity_reduction(2)  # type: ignore[call-arg]
    assert endpoint_count(None) == 1
    assert endpoint_count(2) == 2
    with pytest.raises(ParityReductionError):
        endpoint_count(3)
    with pytest.raises(ParityReductionError):
        endpoint_count(True)
    with pytest.raises(ParityReductionError):
        build_parity_reduction(2, [[], [], []])


def test_nominal_dispatch_for_a_single_vertex_list():
    spec = _spec("sta")
    red = build_parity_reduction(spec.n_vars, [spec.dynamics_gf])
    assert isinstance(red, NominalParityReduction)


# =============================================================================
# The perturbed identities, symbolically
# =============================================================================

@pytest.mark.parametrize("name", sorted(PERTURBED_FACTORIES))
def test_endpoint_swap_identity_holds(name):
    spec = _perturbed_spec(name)
    check = verify_endpoint_swap(spec.dynamics_gf_vertices)

    assert check.holds, f"{name}: {check.message} residuals={check.residuals}"
    assert check.identity == "f_+(-x) = -f_-(x)"


@pytest.mark.parametrize("name", sorted(PERTURBED_FACTORIES))
def test_a_fixed_endpoint_is_not_odd(name):
    """The review finding: naive octant halving has no odd field to stand on."""
    spec = _perturbed_spec(name)
    _, delta0 = PERTURBED_FACTORIES[name]

    for endpoint in (ENDPOINT_PLUS, ENDPOINT_MINUS):
        check = verify_dynamics_odd(spec.dynamics_gf_vertices[endpoint])
        assert not check.holds
        # The residual is the doubled constant disturbance on its channel.
        offending = check.offending
        assert len(offending) == 1
        residual = sp.simplify(check.residual_exprs[offending[0]])
        sign = 1 if endpoint == ENDPOINT_PLUS else -1
        assert abs(float(residual) - sign * 2.0 * delta0) < 1e-12


@pytest.mark.parametrize("name", ["sta", "levant3d", "pidsmc", "levant4d"])
def test_paired_w_identity_and_failure_of_naive_halving(name):
    """W_+(-x) = W_-(x) holds; W_+(-x) = W_+(x) does not.

    Checked on the decrease target the pipeline itself builds. Note that
    compute_lie_derivative_gf returns the Lie derivative dV/dt = grad(V)·f and
    W = -dV/dt, so the two identities are equivalent up to a global sign.
    """
    spec = _perturbed_spec(name)
    V = _diagonal_gf(spec.degree_m, spec.weights_r)
    assert gf_parity(V) == PARITY_EVEN

    symbols = [sp.Symbol(f"x{i + 1}", real=True) for i in range(spec.n_vars)]
    targets = []
    for vertex in spec.dynamics_gf_vertices:
        W = compute_lie_derivative_gf(V, vertex, validate=False)
        assert not W.composite_terms
        targets.append(_to_sympy(W, symbols))

    reflected_plus = targets[ENDPOINT_PLUS].subs(
        {s: -s for s in symbols}, simultaneous=True
    )
    paired_residual = sp.simplify(sp.expand(reflected_plus - targets[ENDPOINT_MINUS]))
    naive_residual = sp.simplify(sp.expand(reflected_plus - targets[ENDPOINT_PLUS]))

    assert paired_residual == 0
    assert naive_residual != 0


def test_naive_halving_mismatch_equals_the_predicted_term():
    """Corollary 10.4 of theory_notes.md, on the Moreno-Osorio V for STA.

    The blocks (s, sigma) and (-s, sigma) differ by exactly
    2*sigma*delta0*dV/dx_c, so the naive identification drops a real
    constraint. delta0 = 1 and c = 2 (the v-channel) here.
    """
    delta0 = 1.0
    spec = ud.create_sta_spec(delta0=delta0)
    V = get_sta_known_lyapunov()
    assert gf_parity(V) == PARITY_EVEN

    x1, x2 = sp.symbols("x1 x2", real=True)
    symbols = [x1, x2]
    V_expr = _to_sympy(V, symbols)

    # W = -dV/dt at the + endpoint.
    W_plus = -_to_sympy(
        compute_lie_derivative_gf(V, spec.dynamics_gf_vertices[ENDPOINT_PLUS], validate=False),
        symbols,
    )
    mismatch = sp.simplify(
        sp.expand(W_plus.subs({s: -s for s in symbols}, simultaneous=True) - W_plus)
    )
    predicted = sp.simplify(2 * 1.0 * delta0 * sp.diff(V_expr, x2))

    for point in [(0.7, -1.3), (-2.0, 0.5), (1.5, 2.5), (-0.25, -0.75)]:
        subs = {x1: point[0], x2: point[1]}
        assert float(mismatch.subs(subs)) == pytest.approx(
            float(predicted.subs(subs)), rel=1e-9, abs=1e-12
        )
        # For the corrected STA V: dV/dx2 = -3*ceil(x1)^(1/2) + 4*x2.
        # The theory note's mismatch is twice that derivative at delta0=1.
        quoted = -6.0 * sp.sign(x1) * sp.Abs(x1) ** 0.5 + 8.0 * x2
        assert float(mismatch.subs(subs)) == pytest.approx(
            float(quoted.subs(subs)), rel=1e-9, abs=1e-12
        )
    assert mismatch != 0


# =============================================================================
# The one-shot plan
# =============================================================================

def test_plan_is_usable_on_a_nominal_benchmark_system():
    spec = _spec("levant3d")
    diag = enumerate_diagonal_terms(spec.degree_m, spec.weights_r)
    _, _, cross = mu_filtered_cross(
        spec.degree_m, spec.weights_r, diag, spec.dynamics_gf
    )
    library = diag + cross

    plan = plan_parity_reduction(
        spec.dynamics_gf, None, terms=library, weights_r=spec.weights_r
    )

    assert plan.usable
    plan.raise_if_unusable()
    assert isinstance(plan.reduction, NominalParityReduction)
    assert plan.endpoint_swap is None
    assert plan.even_terms and all(is_even_term(t) for t in plan.even_terms)
    assert all(not is_even_term(t) for t in plan.odd_terms)
    assert len(plan.even_terms) + len(plan.odd_terms) == len(library)


def test_plan_is_usable_on_a_perturbed_system_and_pairs_endpoints():
    spec = _perturbed_spec("levant3d")
    plan = plan_parity_reduction(spec.dynamics_gf, spec.dynamics_gf_vertices)

    assert plan.usable
    assert isinstance(plan.reduction, PairedParityReduction)
    assert plan.endpoint_swap is not None and plan.endpoint_swap.holds
    assert plan.reduction.n_w_blocks == 2 ** spec.n_vars
    assert plan.reduction.n_v_blocks == 2 ** (spec.n_vars - 1)


def test_plan_is_unusable_when_only_the_endpoint_count_is_given():
    """A count cannot be verified, so the plan refuses to vouch for it."""
    spec = _spec("sta")
    plan = plan_parity_reduction(spec.dynamics_gf, 2)

    assert isinstance(plan.reduction, PairedParityReduction)
    assert plan.endpoint_swap is None
    assert not plan.usable
    with pytest.raises(ParityReductionError, match="endpoint-swap"):
        plan.raise_if_unusable()


def test_plan_is_unusable_for_non_odd_dynamics():
    even_field = GeneralizedForm(
        degree_m=1,
        weights_r=(1,),
        terms=[GFTerm(1.0, (Fraction(1),), ("abs",))],
    )
    plan = plan_parity_reduction([even_field], None)

    assert not plan.usable
    with pytest.raises(ParityReductionError):
        plan.raise_if_unusable()


# =============================================================================
# Block counts for the analytic scaling study (E14a)
# =============================================================================

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_block_counts_used_by_the_scaling_study(n):
    nominal = build_parity_reduction(n, None)
    paired = build_parity_reduction(n, 2)

    assert (nominal.n_v_blocks, nominal.n_w_blocks) == (2 ** (n - 1), 2 ** (n - 1))
    assert (paired.n_v_blocks, paired.n_w_blocks) == (2 ** (n - 1), 2 ** n)
    # Unreduced counts, for the ratio in the scaling table.
    assert nominal.n_octants == 2 ** n
    assert len(paired.full_labels()) == 2 * 2 ** n
