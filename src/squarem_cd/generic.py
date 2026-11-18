"""A generic SQUAREM accelerator for any fixed-point map x <- F(x), in plain NumPy.

This is the algorithm of Varadhan and Roland (2008) as implemented in the SQUAREM package for R, with the alternating
SqS1/SqS2 step lengths and an optional metric. It can be wrapped around any smooth, monotone fixed-point iteration: an EM
or MM algorithm, multiplicative updates for nonnegative matrix factorization, Sinkhorn iterations, or a coordinate
descent epoch.
"""
from dataclasses import dataclass, field

import numpy as np

_STEPS = ("alternate", "sqs1", "sqs2", "sqs3")


@dataclass
class SquaremResult:
    x: np.ndarray
    value: float
    converged: bool
    n_iter: int
    n_fixptfn: int
    n_objfn: int
    n_rejected: int
    history: list = field(default_factory=list)


def squarem(fixptfn, x0, objfn=None, *, step="alternate", inner=None, step_max0=1.0, mstep=4.0, objfn_inc=1.0,
            stabilize=True, tol=1e-7, maxiter=1500, keep_history=False):
    """Accelerate the fixed-point iteration x <- fixptfn(x) with SQUAREM.

    Parameters
    ----------
    fixptfn : callable
        The map; takes and returns a 1-d array of the same length.
    x0 : array_like
        Starting point.
    objfn : callable or None
        An objective that the map decreases (for example a negative log-likelihood). If given, an extrapolation is
        accepted only if it does not increase the objective by more than objfn_inc; otherwise the step after the second
        map evaluation is kept. If None, an extrapolation is accepted when the result is finite.
    step : {"alternate", "sqs1", "sqs2", "sqs3"}
        Step length: SqS1 and SqS2 in alternation (recommended), or one rule throughout.
    inner : callable or None
        Inner product inner(u, w) used for the step lengths (default: the Euclidean one). A problem-specific metric, such
        as u'Hw for a quadratic objective with Hessian H, can work better.
    step_max0, mstep : float
        The step length is confined to [1, step_max]; step_max starts at step_max0, is multiplied by mstep when the step
        reaches it, and divided by mstep after a rejected extrapolation at the maximum.
    objfn_inc : float
        Allowed increase of the objective in the monotonicity check.
    stabilize : bool
        Whether to apply the map once more after the extrapolation (recommended).
    tol : float
        Stop when the Euclidean norm of the change over a cycle is below tol.
    maxiter : int
        Maximum number of cycles.

    Returns
    -------
    SquaremResult with fields x, value (objective at x, or nan), converged, n_iter, n_fixptfn, n_objfn, n_rejected.
    """
    if step not in _STEPS:
        raise ValueError(f"step must be one of {_STEPS}, got {step!r}")
    ip = inner if inner is not None else (lambda u, w: float(np.dot(u, w)))
    x = np.asarray(x0, dtype=float).copy()
    step_max = step_max0
    n_f = 0
    n_obj = 0
    n_rej = 0
    hist = []
    lold = np.nan
    if objfn is not None:
        lold = objfn(x); n_obj += 1
    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        x1 = fixptfn(x); n_f += 1
        q1 = x1 - x
        if np.sqrt(np.dot(q1, q1)) < tol:
            x = x1; converged = True
            break
        x2 = fixptfn(x1); n_f += 1
        q2 = x2 - x1
        v = q2 - q1
        vv = ip(v, v)
        rule = step if step != "alternate" else ("sqs1" if it % 2 == 1 else "sqs2")
        a = 1.0
        if vv > 0:
            qv = ip(q1, v)
            if rule == "sqs1":
                a = -qv / vv
            elif rule == "sqs2":
                a = -ip(q1, q1) / qv if qv < 0 else 1.0
            else:
                a = np.sqrt(ip(q1, q1) / vv)
        if not np.isfinite(a):
            a = 1.0
        a = min(step_max, max(1.0, a))
        xnew = x + 2 * a * q1 + a * a * v
        if stabilize and a != 1.0:
            xnew = fixptfn(xnew); n_f += 1
        accept = np.all(np.isfinite(xnew))
        lnew = np.nan
        if accept and objfn is not None:
            lnew = objfn(xnew); n_obj += 1
            accept = np.isfinite(lnew) and lnew <= lold + objfn_inc
        if not accept:
            n_rej += 1
            xnew = x2
            if objfn is not None:
                lnew = objfn(xnew); n_obj += 1
            if a == step_max:
                step_max = max(step_max0, step_max / mstep)
        elif a == step_max:
            step_max = mstep * step_max
        change = np.sqrt(np.dot(xnew - x, xnew - x))
        x, lold = xnew, lnew
        if keep_history:
            hist.append(lold)
        if change < tol:
            converged = True
            break
    return SquaremResult(x=x, value=float(lold) if objfn is not None else float("nan"), converged=converged,
                         n_iter=it, n_fixptfn=n_f, n_objfn=n_obj, n_rejected=n_rej, history=hist)
