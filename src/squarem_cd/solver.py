"""SquaremCD: coordinate descent with working sets, accelerated by SQUAREM, as a solver for skglm.

The outer loop (working sets, stopping rule, intercept) follows skglm's AndersonCD. Inside each working set, the cyclic
coordinate descent epoch T is accelerated by SQUAREM (Varadhan and Roland, 2008) as in Algorithm 1 of Varadhan,
"Accelerating coordinate descent for sparse regression with squared extrapolation (SQUAREM)":

    two epochs b1 = T(b0), b2 = T(b1); q1 = b1 - b0, v = b2 - 2 b1 + b0;
    step length a from SqS1 / SqS2 in alternation, measured in the metric H = X_A'X_A (inner products of the stored
    differences of Xw, so the metric is free), confined to [1, a_max];
    extrapolation b' = b0 + 2 a q1 + a^2 v (and Xw at b' from the stored Xw);
    one stabilizing epoch b3 = T(b');
    if the objective at b3 exceeds that at b0 by more than objfn_inc, b3 = b2 and a_max is divided by 4 if it bound;
    otherwise a_max is multiplied by 4 if it bound. a_max starts at 1 and is carried across working sets.
"""
import numpy as np
from scipy import sparse
from skglm.solvers.base import BaseSolver
from skglm.solvers.anderson_cd import _cd_epoch, _cd_epoch_sparse
from skglm.solvers.common import construct_grad, construct_grad_sparse, dist_fix_point_cd
from skglm.utils.validation import check_attrs

from ._epochs import _newton_epoch, _newton_epoch_s

_STEPS = ("alternate", "sqs1", "sqs2", "sqs3")


class SquaremCD(BaseSolver):
    """Coordinate descent with working sets and SQUAREM acceleration (a solver for skglm).

    Parameters
    ----------
    max_iter : int, default 50
        Maximum number of outer iterations (working sets).
    max_epochs : int, default 50_000
        Maximum number of coordinate descent epochs per working set.
    p0 : int, default 10
        Minimum size of the working set.
    tol : float, default 1e-4
        Tolerance on the largest violation of the optimality conditions (skglm's criterion).
    ws_strategy : {"subdiff", "fixpoint"}, default "subdiff"
        Score used to build the working sets, as in skglm.
    fit_intercept : bool, default True
        Whether to fit an unpenalized intercept.
    warm_start : bool, default False
        Whether to start from the previous solution (used by skglm's estimators).
    step : {"alternate", "sqs1", "sqs2", "sqs3"}, default "alternate"
        SQUAREM step length: SqS1 and SqS2 in alternation (recommended), or a single rule.
    metric : {"H", "euclidean"}, default "H"
        Inner product for the step length: the metric of X_A'X_A (from the stored Xw, at no cost) or the Euclidean one.
    stabilize : bool, default True
        Whether to take the stabilizing epoch after the extrapolation. Recommended; without it SQUAREM was slower in all
        our tests.
    objfn_inc : float or None, default None
        Allowed increase of the objective in the monotonicity check; None means 1 / n_samples, which is an increase of 1
        on the sum scale (the default of the SQUAREM package), since skglm's datafits are on the mean scale.
    newton_step : bool, default False
        Use the coordinate Newton step (actual curvature, safeguarded) in place of skglm's majorized step. Only for the
        Logistic datafit with the L1 penalty.
    intercept_step : {"newton", "datafit"}, default "newton"
        Update of the intercept: a Newton step sum(raw_grad) / sum(raw_hessian) when the datafit provides them, as in
        skglm's development version; or the datafit's own intercept_update_step.
    verbose : int, default 0
        Amount of printing.

    Attributes (after a solve)
    --------------------------
    n_epochs_, n_cycles_, n_rejected_, n_fallbacks_ : int
        Coordinate descent epochs, SQUAREM cycles, rejected extrapolations, and Newton-step fall-backs.
    """

    _datafit_required_attr = ("get_lipschitz", "gradient_scalar")
    _penalty_required_attr = ("prox_1d",)

    def __init__(self, max_iter=50, max_epochs=50_000, p0=10, tol=1e-4, ws_strategy="subdiff", fit_intercept=True,
                 warm_start=False, step="alternate", metric="H", stabilize=True, objfn_inc=None, newton_step=False,
                 intercept_step="newton", verbose=0):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.tol = tol
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.step = step
        self.metric = metric
        self.stabilize = stabilize
        self.objfn_inc = objfn_inc
        self.newton_step = newton_step
        self.intercept_step = intercept_step
        self.verbose = verbose

    def custom_checks(self, X, y, datafit, penalty):
        check_attrs(datafit, solver=self, required_attr=self._datafit_required_attr,
                    support_sparse=sparse.issparse(X))
        if self.ws_strategy == "subdiff" and not hasattr(penalty, "subdiff_distance"):
            raise AttributeError("Penalty must implement `subdiff_distance` to use ws_strategy='subdiff'.")
        if self.newton_step and (type(datafit).__name__ != "Logistic" or type(penalty).__name__ != "L1"):
            raise ValueError("newton_step=True requires the Logistic datafit and the L1 penalty.")
        if self.step not in _STEPS:
            raise ValueError(f"step must be one of {_STEPS}, got {self.step!r}.")
        if self.metric not in ("H", "euclidean"):
            raise ValueError(f"metric must be 'H' or 'euclidean', got {self.metric!r}.")

    def _solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        if self.ws_strategy not in ("subdiff", "fixpoint"):
            raise ValueError("Unsupported value for ws_strategy: %s" % self.ws_strategy)
        n_samples, n_features = X.shape
        fi = self.fit_intercept
        w = np.zeros(n_features + fi) if w_init is None else np.asarray(w_init, dtype=float).copy()
        Xw = np.zeros(n_samples) if Xw_init is None else np.asarray(Xw_init, dtype=float).copy()
        if len(w) != n_features + fi:
            raise ValueError(f"w should be of size {n_features + fi}, got {len(w)}.")
        is_sparse = sparse.issparse(X)
        if is_sparse:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
            lipschitz = datafit.get_lipschitz_sparse(X.data, X.indptr, X.indices, y)
        else:
            datafit.initialize(X, y)
            lipschitz = datafit.get_lipschitz(X, y)
        pen = penalty.is_penalized(n_features)
        unpen = ~pen
        n_unpen = unpen.sum()
        all_feats = np.arange(n_features)
        inc = 1.0 / n_samples if self.objfn_inc is None else self.objfn_inc
        newton_icpt = (self.intercept_step == "newton" and hasattr(datafit, "raw_grad")
                       and hasattr(datafit, "raw_hessian"))
        y01 = (y + 1.0) / 2.0 if self.newton_step else None
        alpha = penalty.alpha if self.newton_step else 0.0

        def grad(w, Xw, ws):
            if is_sparse:
                return construct_grad_sparse(X.data, X.indptr, X.indices, y, w[:n_features], Xw, datafit, ws)
            return construct_grad(X, y, w[:n_features], Xw, datafit, ws)

        def icpt_step(Xw):
            if newton_icpt:
                return np.sum(datafit.raw_grad(y, Xw)) / np.sum(datafit.raw_hessian(y, Xw))
            return datafit.intercept_update_step(y, Xw)

        def objective(w, Xw):
            return datafit.value(y, w[:n_features], Xw) + penalty.value(w[:n_features])

        def Xmul(w, ws):   # X w (the coefficients outside ws are zero), plus the intercept
            v = X[:, ws] @ w[ws]
            return v + w[-1] if fi else v

        counts = {"epochs": 0, "fallbacks": 0}

        def epoch(w, Xw, ws):   # one cyclic CD epoch on ws, in place, then the intercept
            if self.newton_step:
                if is_sparse:
                    counts["fallbacks"] += _newton_epoch_s(X.data, X.indptr, X.indices, y01, w, Xw, lipschitz, alpha,
                                                           ws.astype(np.int64), n_samples)
                else:
                    counts["fallbacks"] += _newton_epoch(X, y01, w, Xw, lipschitz, alpha, ws.astype(np.int64))
            elif is_sparse:
                _cd_epoch_sparse(X.data, X.indptr, X.indices, y, w[:n_features], Xw, lipschitz, datafit, penalty, ws)
            else:
                _cd_epoch(X, y, w[:n_features], Xw, lipschitz, datafit, penalty, ws)
            if fi:
                old = w[-1]
                w[-1] -= icpt_step(Xw)
                Xw += w[-1] - old
            counts["epochs"] += 1

        def violation_ws(w, Xw, ws):
            grad_ws = grad(w, Xw, ws)
            if self.ws_strategy == "subdiff":
                opt_ws = penalty.subdiff_distance(w[:n_features], grad_ws, ws)
            else:
                opt_ws = dist_fix_point_cd(w[:n_features], grad_ws, lipschitz[ws], datafit, penalty, ws)
            return np.max(opt_ws)

        obj_out = []
        stop_crit = np.inf
        a_max = 1.0
        kcyc = 0
        self.n_cycles_ = 0
        self.n_rejected_ = 0
        for t in range(self.max_iter):
            g = grad(w, Xw, all_feats)
            if self.ws_strategy == "subdiff":
                opt = penalty.subdiff_distance(w[:n_features], g, all_feats)
            else:
                opt = dist_fix_point_cd(w[:n_features], g, lipschitz, datafit, penalty, all_feats)
            icpt_opt = abs(icpt_step(Xw)) if fi else 0.0
            stop_crit = max(np.max(opt), icpt_opt)
            if self.verbose:
                print(f"Iteration {t + 1}: stopping criterion {stop_crit:.2e}")
            if stop_crit <= self.tol:
                break
            ws_size = max(min(self.p0 + n_unpen, n_features),
                          min(2 * penalty.generalized_support(w[:n_features]).sum() - n_unpen, n_features))
            opt[unpen] = np.inf
            opt[penalty.generalized_support(w[:n_features])] = np.inf
            ws = np.argpartition(opt, -ws_size)[-ws_size:]
            wsi = np.append(ws, n_features) if fi else ws
            tol_in = self.tol if ws_size == n_features else 0.3 * stop_crit

            def inner_done(wc, Xwc):
                v = violation_ws(wc, Xwc, ws)
                return v <= tol_in if ws_size == n_features else v < tol_in

            # the SQUAREM-accelerated inner loop
            w0, Xw0 = w.copy(), Xw.copy()
            l0 = objective(w0, Xw0)
            n_ep0 = counts["epochs"]
            last_check = None
            while True:
                w1, Xw1 = w0.copy(), Xw0.copy(); epoch(w1, Xw1, ws)
                if last_check is None:   # first epoch of the working set: check, as skglm does
                    last_check = counts["epochs"]
                    if inner_done(w1, Xw1):
                        w0, Xw0 = w1, Xw1
                        break
                w2, Xw2 = w1.copy(), Xw1.copy(); epoch(w2, Xw2, ws)
                q1 = w1[wsi] - w0[wsi]
                v = w2[wsi] - 2 * w1[wsi] + w0[wsi]
                if self.metric == "H":
                    d1 = Xw1 - Xw0; vv = Xw2 - 2 * Xw1 + Xw0
                else:
                    d1 = q1; vv = v
                kcyc += 1
                sv2 = vv @ vv
                a = 1.0
                if sv2 > 0:
                    rule = self.step if self.step != "alternate" else ("sqs1" if kcyc % 2 == 1 else "sqs2")
                    dv = d1 @ vv
                    if rule == "sqs1":
                        a = -dv / sv2
                    elif rule == "sqs2":
                        a = -(d1 @ d1) / dv if dv < 0 else 1.0
                    else:
                        a = np.sqrt((d1 @ d1) / sv2)
                if not np.isfinite(a):
                    a = 1.0
                a = min(a_max, max(1.0, a))
                wx = w0.copy()
                wx[wsi] = w0[wsi] + 2 * a * q1 + a * a * v
                if a > 10:   # the a^2 term amplifies rounding errors in the stored Xw: recompute it
                    Xwx = Xmul(wx, ws)
                else:
                    Xwx = Xw0 + 2 * a * (Xw1 - Xw0) + a * a * (Xw2 - 2 * Xw1 + Xw0)
                if self.stabilize:
                    w3, Xw3 = wx, Xwx; epoch(w3, Xw3, ws)
                else:
                    w3, Xw3 = wx, Xwx
                l3 = objective(w3, Xw3)
                self.n_cycles_ += 1
                if not np.isfinite(l3) or l3 > l0 + inc:
                    self.n_rejected_ += 1
                    w3, Xw3 = w2, Xw2
                    l3 = objective(w3, Xw3)
                    if a == a_max:
                        a_max = max(1.0, a_max / 4)
                elif a == a_max:
                    a_max = 4 * a_max
                w0, Xw0, l0 = w3, Xw3, l3
                if counts["epochs"] - last_check >= 10:
                    last_check = counts["epochs"]
                    if inner_done(w0, Xw0):
                        break
                if counts["epochs"] - n_ep0 >= self.max_epochs:
                    break
            w, Xw = w0, Xw0
            p_obj = objective(w, Xw)
            obj_out.append(p_obj)
            if self.verbose:
                print(f"  {counts['epochs'] - n_ep0} epochs on {ws_size} features, objective {p_obj:.10f}")
        self.n_epochs_ = counts["epochs"]
        self.n_fallbacks_ = counts["fallbacks"]
        return w, np.array(obj_out), stop_crit
