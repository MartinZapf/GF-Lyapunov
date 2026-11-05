"""Direct MOSEK Task-API backend for the joint V+W SOS design.

Encodes the SAME SDP as ``JointSOSOptimizer.find_lyapunov_with_dynamics``
(scalar coefficient vector α, one PSD Gram block per octant for V and per
octant-vertex for W, linear monomial-matching rows) directly through MOSEK's
low-level Task API, bypassing CVXPY canonicalization entirely. Motivation:
at n = 5 (Levant-5D, 32 octants, 210/126-row Grams) CVXPY's
ConeMatrixStuffing exhausted 121 GB before MOSEK ever started; the same
problem encoded natively needs the Gram entries (≈ 0.97 M scalars) plus a
sparse constraint matrix.

Conventions (pinned by ``tests/test_mosek_direct.py::test_bara_convention``):

- ``build_gram_constraints`` (joint_sos.py) lists BOTH ordered pairs (i, j)
  and (j, i) for i ≠ j, so the CVXPY sum ``Σ Q[i,j]`` counts off-diagonals
  twice. MOSEK's ⟨A, X⟩ for a symmetric coefficient matrix specified on the
  lower triangle doubles off-diagonal entries as well, so after deduplicating
  each pair list to ``{(max(i,j), min(i,j))}`` the barA weight is **1.0
  everywhere** — no 2×/0.5× factors.
- ``Task.getbarxj`` returns the lower triangle in column-major packed order
  ((0,0), (1,0), …, (d−1,0), (1,1), …); see ``unpack_lower_triangle``.
- ``putbarablocktriplet`` takes the 5-argument MOSEK-11 form
  (subi, subj, subk, subl, val) with subk ≥ subl.

Trust boundary: identical to the CVXPY path — this is the DESIGN side; a
returned α is only a proposal and every stability claim passes the strict
gate (``certify_discovered_gf``). A ``solsta.unknown`` solution is accepted
only if the recovered Grams pass a PSD + residual screening (the analogue of
accepting ``OPTIMAL_INACCURATE``). Every solve recomputes the per-row
residual from the recovered Grams, so a convention error would surface as an
O(1) ``gram_residual_l1`` immediately.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Configure MOSEK license (portable): only set if user has not configured it.
if 'MOSEKLM_LICENSE_FILE' not in os.environ:
    _repo_root = Path(__file__).resolve().parents[1]
    _license_path = _repo_root / 'mosek.lic'
    if _license_path.exists():
        os.environ['MOSEKLM_LICENSE_FILE'] = str(_license_path)

try:
    import mosek

    MOSEK_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without MOSEK
    MOSEK_AVAILABLE = False


@dataclass
class GramRow:
    """One monomial-matching row: a_vals·α − ⟨A, Q⟩ = rhs.

    ``pairs_k/pairs_l`` index the LOWER triangle of the block's Gram
    (k ≥ l, deduplicated); the barA weight is 1.0 at every listed slot.

    ``extra`` (WP P5-04) lets ONE row touch several PSD blocks, which the
    y-parity decomposition needs: splitting a dense Gram into its parity
    classes turns one coefficient-matching row into a row over the within-class
    pair sums of ALL classes at once. Each entry is
    ``(block_offset, pairs_k, pairs_l)`` with ``block_offset`` counted from the
    block that OWNS the row, so a group of sub-blocks stays relocatable as a
    unit (they are always emitted contiguously). Empty on every dense block, so
    the historical encoding is byte-identical.
    """
    a_cols: np.ndarray   # alpha column indices (int), possibly empty
    a_vals: np.ndarray   # matching M-row values at those columns
    rhs: float           # scaled epsilon on pure-power V rows, else 0.0
    pairs_k: np.ndarray  # lower-tri row indices (k >= l)
    pairs_l: np.ndarray  # lower-tri col indices
    extra: Tuple[Tuple[int, np.ndarray, np.ndarray], ...] = ()


@dataclass
class GramBlock:
    """One PSD barvar: an (octant, V) or (octant, vertex, W) SOS block."""
    dim: int
    name: str
    rows: List[GramRow] = field(default_factory=list)


@dataclass
class DirectSolveOutcome:
    status: str                 # optimal | unknown_accepted | infeasible | unknown | error
    alpha: Optional[np.ndarray]  # SCALED alpha (length n_terms) or None
    objective: Optional[float]
    solve_time: float
    gram_min_eig: Optional[float] = None      # worst block min eigenvalue
    gram_residual_l1: Optional[float] = None  # worst row |a·α − rhs − ⟨A,Q⟩|
    grams: Optional[List[np.ndarray]] = None  # recovered Grams (diagnostics on)
    message: str = ""


@dataclass
class PolySolveOutcome:
    """Outcome of ONE single-polynomial SOS feasibility solve (WP P2-10).

    ``status`` is colon-free and drawn from ``optimal | unknown | infeasible |
    error | unavailable`` so it types straight through
    ``failure_typing._classify_attempt`` when it lands in an attempt string.
    ``gram`` is a PROPOSAL: acceptance is ``joint_sos._validate_gram_certificate``
    (PSD margin + equality residual charged against ε), never ``status``.
    """
    status: str
    gram: Optional[np.ndarray]
    solve_time: float
    message: str = ""


def unpack_lower_triangle(packed: np.ndarray, dim: int) -> np.ndarray:
    """Symmetric matrix from MOSEK's column-major packed lower triangle."""
    Q = np.zeros((dim, dim))
    idx = 0
    for col in range(dim):
        n = dim - col
        Q[col:, col] = packed[idx:idx + n]
        idx += n
    Q = Q + Q.T - np.diag(np.diag(Q))
    return Q


def blocks_from_assembly(
    M: np.ndarray,
    monomials: Sequence[Tuple[int, ...]],
    gram_map: Dict[Tuple[int, ...], List[Tuple[int, int]]],
    eps_rows: Sequence[int],
    scaled_epsilon: float,
    dim: int,
    name: str,
    zero_pairs: Optional[Dict[Tuple[int, ...], List[Tuple[int, int]]]] = None,
) -> GramBlock:
    """Build one GramBlock from the matching matrix + pair map.

    Skips rows that are trivially 0 == 0 (all-zero M-row, no Gram pairs,
    zero rhs) — CVXPY materializes those as constraints; they are vacuous.

    ``zero_pairs`` (WP P2-09, from ``joint_sos.build_gram_completion_map``)
    holds the basis products that land OUTSIDE ``monomials``, the block's
    target support. Each gets an explicit ``⟨A, Q⟩ = 0`` row, because a
    product monomial with no row leaves its Gram entries free: the solver can
    park arbitrary mass there and the block stops being a certificate of the
    target (measured: ``power_rate``'s W ``Q[2,2] = 6.16e+04``, P2-04). These
    rows are vacuous — and the map is empty — whenever the target support
    already covers every product, which is the homogeneous even-degree case.
    """
    eps_set = set(eps_rows)
    block = GramBlock(dim=dim, name=name)
    for r, mono in enumerate(monomials):
        a_row = M[r, :]
        cols = np.nonzero(a_row)[0]
        pairs = gram_map.get(mono, [])
        rhs = scaled_epsilon if r in eps_set else 0.0
        if cols.size == 0 and not pairs and rhs == 0.0:
            continue
        if pairs:
            dedup = sorted({(max(i, j), min(i, j)) for i, j in pairs})
            pk = np.fromiter((p[0] for p in dedup), dtype=np.int32,
                             count=len(dedup))
            pl = np.fromiter((p[1] for p in dedup), dtype=np.int32,
                             count=len(dedup))
        else:
            pk = np.empty(0, dtype=np.int32)
            pl = np.empty(0, dtype=np.int32)
        block.rows.append(GramRow(
            a_cols=cols.astype(np.int32), a_vals=a_row[cols].astype(float),
            rhs=float(rhs), pairs_k=pk, pairs_l=pl))
    for _mono, pairs in sorted((zero_pairs or {}).items()):
        if not pairs:
            continue
        dedup = sorted({(max(i, j), min(i, j)) for i, j in pairs})
        block.rows.append(GramRow(
            a_cols=np.empty(0, dtype=np.int32), a_vals=np.empty(0, dtype=float),
            rhs=0.0,
            pairs_k=np.fromiter((p[0] for p in dedup), dtype=np.int32,
                                count=len(dedup)),
            pairs_l=np.fromiter((p[1] for p in dedup), dtype=np.int32,
                                count=len(dedup))))
    return block


def _pair_sum(Q: np.ndarray, pk: np.ndarray, pl: np.ndarray) -> float:
    """⟨A, Q⟩ for a deduplicated lower-triangle pair list (off-diagonals ×2)."""
    if not pk.size:
        return 0.0
    q = Q[pk, pl]
    diag = pk == pl
    return float(np.sum(q[diag]) + 2.0 * np.sum(q[~diag]))


def blocks_from_split(split: "Any") -> List[GramBlock]:
    """MOSEK blocks for one y-parity-decomposed Gram block (WP P5-04).

    ``split`` is a ``yparity.SplitBlock``. One PSD barvar per parity class, in
    the canonical ``split.class_order``; every row is attached to the FIRST
    sub-block of the group and reaches the others through ``GramRow.extra``
    with offsets relative to it. Sub-block names are ``<dense name>#p<class>``,
    so a blocked run stays diffable against the dense one.

    Soundness is the group-averaging argument in ``yparity``: a block-diagonal
    Gram is a dense Gram (so nothing is accepted that the dense encoding would
    reject), and the parity projection of any dense solution is a
    block-diagonal one of the same α (so nothing feasible is lost). Both halves
    are pinned MOSEK-free in ``tests/test_yparity.py``.
    """
    order = split.class_order
    pos = {key: i for i, key in enumerate(order)}
    blocks = [
        GramBlock(dim=int(split.sub_dims[key]),
                  name=f"{split.name}#p{''.join(str(int(e)) for e in key)}")
        for key in order
    ]
    if not blocks:
        return blocks
    for row in split.rows:
        own_k = np.empty(0, dtype=np.int32)
        own_l = np.empty(0, dtype=np.int32)
        extra: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for key in order:
            pair = row.class_pairs.get(key)
            if pair is None or not len(pair[0]):
                continue
            if pos[key] == 0:
                own_k, own_l = pair
            else:
                extra.append((pos[key], pair[0], pair[1]))
        blocks[0].rows.append(GramRow(
            a_cols=row.a_cols, a_vals=row.a_vals, rhs=float(row.rhs),
            pairs_k=own_k, pairs_l=own_l, extra=tuple(extra)))
    return blocks


def _block_diagnostics(blocks: List[GramBlock], barx: List[np.ndarray],
                       alpha: np.ndarray) -> Tuple[float, float]:
    """(worst min eigenvalue, worst row residual) over all blocks."""
    worst_eig = np.inf
    worst_res = 0.0
    grams = [unpack_lower_triangle(packed, b.dim)
             for b, packed in zip(blocks, barx)]
    for Q in grams:
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(Q).min()))
    for j, b in enumerate(blocks):
        for row in b.rows:
            a_part = float(row.a_vals @ alpha[row.a_cols]) if row.a_cols.size else 0.0
            aq = _pair_sum(grams[j], row.pairs_k, row.pairs_l)
            for off, pk, pl in row.extra:
                aq += _pair_sum(grams[j + off], pk, pl)
            worst_res = max(worst_res, abs(a_part - row.rhs - aq))
    if not np.isfinite(worst_eig):
        worst_eig = 0.0
    return worst_eig, worst_res


def solve_joint_sos_direct(
    *,
    n_terms: int,
    blocks: List[GramBlock],
    diag_indices: Sequence[int],
    scaled_epsilon: float,
    coefficient_scale: float,
    l1_weights_cross: Optional[Tuple[Sequence[int], Sequence[float]]] = None,
    minimize_gram_trace: bool = False,
    num_threads: int = 0,
    timeout: Optional[float] = None,
    mosek_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    compute_diagnostics: bool = True,
) -> DirectSolveOutcome:
    """Solve the assembled joint SOS SDP through the MOSEK Task API.

    ``l1_weights_cross``: optional (cross_indices, scaled_weights) for the
    weighted-ℓ₁ objective Σ wⱼ·|αⱼ| over the cross terms, encoded via
    auxiliary epigraph variables tⱼ ≥ ±αⱼ. Feasibility otherwise.

    ``minimize_gram_trace`` (WP P5-05, opt-in, default off): ADD Σ_j tr Q_j
    over the PSD blocks to that objective. ⟨I, Q⟩ = tr Q is encoded with one
    identity ``appendsparsesymmat`` per distinct block dimension, attached with
    ``putbarcj``. On the PSD cone the trace is the nuclear norm, so this is the
    convex rank surrogate; it is bounded below by 0 there and cannot make a
    feasible problem unbounded.
    """
    if not MOSEK_AVAILABLE:  # pragma: no cover
        return DirectSolveOutcome(status="error", alpha=None, objective=None,
                                  solve_time=0.0, message="mosek not importable")
    t0 = time.time()
    cross_idx: List[int] = []
    cross_w: List[float] = []
    if l1_weights_cross is not None:
        cross_idx = list(l1_weights_cross[0])
        cross_w = [float(w) for w in l1_weights_cross[1]]
        assert len(cross_idx) == len(cross_w)
    n_cross = len(cross_idx)
    n_rows = sum(len(b.rows) for b in blocks)
    numvar = n_terms + n_cross
    numcon = n_rows + 2 * n_cross

    try:
        with mosek.Env() as env, env.Task(0, 0) as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log,
                                lambda s: print(s, end="", flush=True))
            task.appendvars(numvar)
            task.appendbarvars([b.dim for b in blocks])
            task.appendcons(numcon)

            # --- scalar variable bounds -------------------------------------
            bk = [mosek.boundkey.fr] * numvar
            bl = [0.0] * numvar
            bu = [0.0] * numvar
            for pos, i in enumerate(diag_indices):
                if pos == 0:
                    bk[i] = mosek.boundkey.fx
                    bl[i] = bu[i] = coefficient_scale
                else:
                    bk[i] = mosek.boundkey.lo
                    bl[i] = scaled_epsilon
            task.putvarboundlist(list(range(numvar)), bk, bl, bu)

            # --- linear rows + barA triplets ---------------------------------
            sub, ptrb, ptre, asub, aval = [], [], [], [], []
            con_bk, con_bl, con_bu = [], [], []
            tri_i, tri_j, tri_k, tri_l, tri_v = [], [], [], [], []
            row_id = 0
            for j, b in enumerate(blocks):
                for row in b.rows:
                    sub.append(row_id)
                    ptrb.append(len(asub))
                    asub.extend(int(c) for c in row.a_cols)
                    aval.extend(float(v) for v in row.a_vals)
                    ptre.append(len(asub))
                    con_bk.append(mosek.boundkey.fx)
                    con_bl.append(row.rhs)
                    con_bu.append(row.rhs)
                    if row.pairs_k.size:
                        npairs = row.pairs_k.size
                        tri_i.extend([row_id] * npairs)
                        tri_j.extend([j] * npairs)
                        tri_k.extend(int(k) for k in row.pairs_k)
                        tri_l.extend(int(l) for l in row.pairs_l)
                        tri_v.extend([-1.0] * npairs)
                    # P5-04: the same row over the other parity sub-blocks of
                    # this group. Offsets are relative to the owning block.
                    for off, pk, pl in row.extra:
                        if not len(pk):
                            continue
                        tri_i.extend([row_id] * len(pk))
                        tri_j.extend([j + int(off)] * len(pk))
                        tri_k.extend(int(k) for k in pk)
                        tri_l.extend(int(l) for l in pl)
                        tri_v.extend([-1.0] * len(pk))
                    row_id += 1
            # epigraph rows: t_c − α_c ≥ 0 and t_c + α_c ≥ 0
            for c_pos, c in enumerate(cross_idx):
                t_var = n_terms + c_pos
                for sgn in (-1.0, +1.0):
                    sub.append(row_id)
                    ptrb.append(len(asub))
                    asub.extend([t_var, int(c)])
                    aval.extend([1.0, sgn])
                    ptre.append(len(asub))
                    con_bk.append(mosek.boundkey.lo)
                    con_bl.append(0.0)
                    con_bu.append(0.0)
                    row_id += 1
            task.putarowlist(sub, ptrb, ptre, asub, aval)
            task.putconboundlist(list(range(numcon)), con_bk, con_bl, con_bu)
            if tri_i:
                task.putbarablocktriplet(tri_i, tri_j, tri_k, tri_l, tri_v)

            # --- objective ----------------------------------------------------
            if n_cross:
                task.putclist([n_terms + p for p in range(n_cross)], cross_w)
            if minimize_gram_trace and blocks:
                # ⟨I_d, Q⟩ = tr Q. One identity symmat per DISTINCT dimension,
                # reused across blocks — MOSEK's symmetric-matrix store is
                # global, so the 2^n blocks of one design cost at most a
                # handful of entries.
                eye_idx: Dict[int, int] = {}
                for b in blocks:
                    if b.dim not in eye_idx:
                        diag = list(range(b.dim))
                        eye_idx[b.dim] = task.appendsparsesymmat(
                            b.dim, diag, diag, [1.0] * b.dim)
                for j, b in enumerate(blocks):
                    task.putbarcj(j, [eye_idx[b.dim]], [1.0])
            task.putobjsense(mosek.objsense.minimize)

            # --- parameters ---------------------------------------------------
            task.putintparam(mosek.iparam.num_threads, int(num_threads))
            if timeout is not None:
                task.putdouparam(mosek.dparam.optimizer_max_time, float(timeout))
            for name, value in (mosek_params or {}).items():
                task.putparam(str(name), str(value))

            task.optimize()
            if verbose:
                task.solutionsummary(mosek.streamtype.log)
            solsta = task.getsolsta(mosek.soltype.itr)
            prosta = task.getprosta(mosek.soltype.itr)

            if solsta == mosek.solsta.prim_infeas_cer or \
                    prosta == mosek.prosta.prim_infeas:
                return DirectSolveOutcome(
                    status="infeasible", alpha=None, objective=None,
                    solve_time=time.time() - t0,
                    message=f"primal infeasibility certificate ({solsta})")
            if solsta == mosek.solsta.dual_infeas_cer:
                return DirectSolveOutcome(
                    status="error", alpha=None, objective=None,
                    solve_time=time.time() - t0,
                    message="dual infeasibility certificate (unbounded?)")

            xx = np.array(task.getxx(mosek.soltype.itr))
            alpha = xx[:n_terms]
            objective = float(task.getprimalobj(mosek.soltype.itr))
            min_eig = residual = grams = None
            if compute_diagnostics or solsta != mosek.solsta.optimal:
                barx = [np.array(task.getbarxj(mosek.soltype.itr, j))
                        for j in range(len(blocks))]
                min_eig, residual = _block_diagnostics(blocks, barx, alpha)
                grams = [unpack_lower_triangle(p, b.dim)
                         for p, b in zip(barx, blocks)]

            if solsta == mosek.solsta.optimal:
                return DirectSolveOutcome(
                    status="optimal", alpha=alpha, objective=objective,
                    solve_time=time.time() - t0,
                    gram_min_eig=min_eig, gram_residual_l1=residual,
                    grams=grams, message="MOSEK optimal")
            # solsta unknown: accept only if the recovered Grams screen clean —
            # the analogue of the CVXPY path accepting OPTIMAL_INACCURATE. The
            # strict gate stays the authority either way.
            # NOTE: these screening thresholds are backend-LOCAL — the CVXPY
            # path forwards the solver's own status without a Gram screen, so
            # the two backends apply different design-side acceptance criteria.
            # The asymmetry is harmless by construction: both paths emit
            # PROPOSALS only, and nothing becomes a claim before
            # certify_discovered_gf (strict, vertex-aware) and, for published
            # certificates, the exact-rational gate.
            eig_tol = 1e-7 * max(1.0, coefficient_scale)
            res_tol = 1e-5 * max(1.0, coefficient_scale)
            if min_eig is not None and min_eig >= -eig_tol and residual <= res_tol:
                return DirectSolveOutcome(
                    status="unknown_accepted", alpha=alpha, objective=objective,
                    solve_time=time.time() - t0,
                    gram_min_eig=min_eig, gram_residual_l1=residual,
                    grams=grams,
                    message=f"MOSEK {solsta}; Grams screen clean "
                            f"(min_eig={min_eig:.2e}, residual={residual:.2e})")
            return DirectSolveOutcome(
                status="unknown", alpha=None, objective=None,
                solve_time=time.time() - t0,
                gram_min_eig=min_eig, gram_residual_l1=residual,
                message=f"MOSEK {solsta}; Gram screening failed "
                        f"(min_eig={min_eig}, residual={residual})")
    except mosek.Error as exc:  # licensing, sizes, parameter names, ...
        return DirectSolveOutcome(
            status="error", alpha=None, objective=None,
            solve_time=time.time() - t0,
            message=f"MOSEK error: {exc}")


#: Interior-point accuracy for the WITNESS problem (WP P2-10). MOSEK's defaults
#: (1e-8 pfeas/dfeas) leave an equality residual around 2e-10 on a Gram this
#: small, and Proposition 7 charges ‖E‖₁ against ε/2 — at ε = 1e-6 that is 4e-4
#: of the budget spent on solver slop alone, and it grows with the instance.
#: CVXPY's MOSEK interface reaches ~1e-16 because its canonicalization rescales
#: the problem; the Task API sees it raw, so the accuracy target is set here
#: instead. MEASURED on the 12 nominal E1 rows at eps = 1e-6: worst PRIMARY
#: witness charge/budget drops from 2.42e-01 to 1.46e-02, the rows retried under
#: D1's 0.1 trigger drop from 5 to 0, and the three infeasible float witnesses
#: (fc_cta, continuous_twisting, levant4d) disappear. See
#: artifacts/analysis_2026-09-02/exact_solver_policy_direct/README.md and
#: NEGATIVES.md N19: those infeasibilities are an ACCURACY artefact, not the
#: CVXPY-encoding artefact of N7/N9/N10.
#: This is an accuracy target, not a formulation change: the encoding, the
#: ε-margin and the a-posteriori Gram acceptance are untouched.
POLY_SOS_MOSEK_PARAMS: Dict[str, str] = {
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": "1e-12",
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": "1e-12",
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": "1e-12",
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": "1e-13",
}


def solve_poly_sos_direct(
    *,
    target: Dict[Tuple[int, ...], float],
    gram_map: Dict[Tuple[int, ...], List[Tuple[int, int]]],
    n_basis: int,
    num_threads: int = 0,
    timeout: Optional[float] = None,
    mosek_params: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> PolySolveOutcome:
    """Single-polynomial SOS feasibility through the MOSEK Task API (WP P2-10).

    Poses EXACTLY the problem ``joint_sos._numeric_poly_is_sos`` poses in CVXPY:
    one PSD Gram block ``Q`` of dimension ``n_basis``, one equality row per
    monomial reachable by the basis or present in ``target``,

        Σ_{(i,j) ∈ gram_map[mono]} Q[i, j] == target[mono],

    objective ``Minimize 0``. ``gram_map`` is the caller's map from a monomial to
    the ORDERED index pairs of the basis product, so off-diagonals appear twice;
    after deduplicating to the lower triangle MOSEK's ⟨A, X⟩ doubles them again,
    so the barA weight is 1.0 everywhere (the convention pinned at the top of
    this module). There are no scalar variables: this is the pure Gram feasibility
    problem, not the joint design SDP.

    ``target`` must already carry the caller's ε-margin; nothing is subtracted
    here, so the epsilon/margin semantics are the CVXPY branch's verbatim.

    ``mosek_params`` OVERRIDES entries of ``POLY_SOS_MOSEK_PARAMS`` (the
    tightened interior-point accuracy target); pass ``{}`` to keep it, and an
    explicit key to change one setting.

    Trust boundary, unchanged: the returned Gram is a PROPOSAL. The caller runs
    ``_validate_gram_certificate`` on it and, for a published certificate, the
    exact rational gate runs after that. A ``solsta`` string is never proof.
    """
    if not MOSEK_AVAILABLE:  # pragma: no cover - environment without MOSEK
        return PolySolveOutcome(status="unavailable", gram=None, solve_time=0.0,
                                message="mosek not importable")
    if int(n_basis) <= 0:
        return PolySolveOutcome(status="error", gram=None, solve_time=0.0,
                                message="empty basis")
    t0 = time.time()

    # --- rows: one per monomial the basis can reach or the target names --------
    monos = sorted(set(gram_map) | set(target))
    row_rhs: List[float] = []
    tri_i: List[int] = []
    tri_j: List[int] = []
    tri_k: List[int] = []
    tri_l: List[int] = []
    tri_v: List[float] = []
    row_id = 0
    for mono in monos:
        pairs = gram_map.get(mono, [])
        coeff = float(target.get(mono, 0.0))
        if not pairs:
            if coeff != 0.0:
                # target needs a monomial the basis cannot produce -> not SOS
                return PolySolveOutcome(
                    status="infeasible", gram=None, solve_time=time.time() - t0,
                    message="monomial_outside_basis")
            continue  # vacuous 0 == 0 row
        dedup = sorted({(max(i, j), min(i, j)) for i, j in pairs})
        tri_i.extend([row_id] * len(dedup))
        tri_j.extend([0] * len(dedup))
        tri_k.extend(int(p[0]) for p in dedup)
        tri_l.extend(int(p[1]) for p in dedup)
        tri_v.extend([1.0] * len(dedup))
        row_rhs.append(coeff)
        row_id += 1

    if row_id == 0:
        return PolySolveOutcome(status="error", gram=None, solve_time=time.time() - t0,
                                message="no matching rows")

    try:
        with mosek.Env() as env, env.Task(0, 0) as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log,
                                lambda s: print(s, end="", flush=True))
            task.appendbarvars([int(n_basis)])
            task.appendcons(row_id)
            task.putconboundlist(
                list(range(row_id)),
                [mosek.boundkey.fx] * row_id,
                row_rhs, row_rhs,
            )
            task.putbarablocktriplet(tri_i, tri_j, tri_k, tri_l, tri_v)
            task.putobjsense(mosek.objsense.minimize)

            task.putintparam(mosek.iparam.num_threads, int(num_threads))
            if timeout is not None:
                task.putdouparam(mosek.dparam.optimizer_max_time, float(timeout))
            params = dict(POLY_SOS_MOSEK_PARAMS)
            params.update(mosek_params or {})
            for name, value in params.items():
                task.putparam(str(name), str(value))

            task.optimize()
            if verbose:
                task.solutionsummary(mosek.streamtype.log)
            solsta = task.getsolsta(mosek.soltype.itr)
            prosta = task.getprosta(mosek.soltype.itr)

            if solsta == mosek.solsta.prim_infeas_cer or \
                    prosta == mosek.prosta.prim_infeas:
                return PolySolveOutcome(
                    status="infeasible", gram=None, solve_time=time.time() - t0,
                    message=f"primal infeasibility certificate ({solsta})")
            if solsta == mosek.solsta.dual_infeas_cer:
                return PolySolveOutcome(
                    status="error", gram=None, solve_time=time.time() - t0,
                    message="dual infeasibility certificate")

            packed = np.array(task.getbarxj(mosek.soltype.itr, 0))
            gram = unpack_lower_triangle(packed, int(n_basis))
            # solsta.optimal and every other solution-bearing status hand the
            # SAME object back: a proposal. The analogue of the CVXPY branch
            # accepting OPTIMAL and OPTIMAL_INACCURATE alike, with
            # _validate_gram_certificate as the only acceptance test.
            status = "optimal" if solsta == mosek.solsta.optimal else "unknown"
            return PolySolveOutcome(
                status=status, gram=gram, solve_time=time.time() - t0,
                message=f"MOSEK {solsta}")
    except mosek.Error as exc:  # licensing, sizes, parameter names, ...
        return PolySolveOutcome(
            status="error", gram=None, solve_time=time.time() - t0,
            message=f"MOSEK error {exc}")
