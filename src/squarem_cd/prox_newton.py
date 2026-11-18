"""ProxNewtonSquarem: skglm's proximal Newton solver for sparse generalized linear models, with SQUAREM-H in the inner
solve of each Newton step.

This module is derived from skglm/solvers/prox_newton.py of skglm 0.5 (https://github.com/scikit-learn-contrib/skglm),
Copyright (c) 2023, scikit-learn-contrib (the skglm developers), distributed under the BSD 3-Clause License, whose text is reproduced in the
LICENSE file of this package. Changes (Ravi Varadhan, 2026): the inner coordinate descent solve of each Newton step, which
skglm caps at 20 epochs, can use SQUAREM-H on the same epochs (inner="squarem", the default here): cycles of two epochs,
an extrapolation with SqS1/SqS2 step lengths in alternation, measured in the metric of the Hessian of the quadratic model
(inner products of the stored X delta weighted by the Hessian weights), and a stabilizing epoch, with a check that the
model value decreases; the same cap of 20 epochs and the same stopping rule. inner="cd" gives skglm's loop. Epoch counts
are kept in n_pn and n_epochs.
"""

import warnings

import numpy as np
from numba import njit
from scipy.sparse import issparse
from skglm.solvers.base import BaseSolver
from skglm.solvers.common import dist_fix_point_cd

from sklearn.exceptions import ConvergenceWarning
from skglm.utils.sparse_ops import _sparse_xj_dot

EPS_TOL = 0.3
MAX_CD_ITER = 20
MAX_BACKTRACK_ITER = 20


class ProxNewtonSquarem(BaseSolver):
    """Prox Newton solver combined with working sets.

    p0 : int, default 10
        Minimum number of features to be included in the working set.

    max_iter : int, default 20
        Maximum number of outer iterations.

    max_pn_iter : int, default 1000
        Maximum number of prox Newton iterations on each subproblem.

    tol : float, default 1e-4
        Tolerance for convergence.

    ws_strategy : ('subdiff'|'fixpoint'), optional
        The score used to build the working set.

    fit_intercept : bool, default True
        If ``True``, fits an unpenalized intercept.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.

    inner : {"squarem", "cd"}, default "squarem"
        Inner solver of each Newton step: SQUAREM-H on the coordinate descent epochs, or skglm's plain epochs.

    References
    ----------
    .. [1] Massias, M. and Vaiter, S. and Gramfort, A. and Salmon, J.
        "Dual Extrapolation for Sparse Generalized Linear Models", JMLR, 2020,
        https://arxiv.org/abs/1907.05830
        code: https://github.com/mathurinm/celer

    .. [2] Johnson, T. B. and Guestrin, C.
        "Blitz: A principled meta-algorithm for scaling sparse optimization",
        ICML, 2015.
        https://proceedings.mlr.press/v37/johnson15.html
        code: https://github.com/tbjohns/BlitzL1
    """

    _datafit_required_attr = ("raw_grad", "raw_hessian")
    _penalty_required_attr = ("prox_1d",)

    def __init__(self, p0=10, max_iter=20, max_pn_iter=1000, tol=1e-4,
                 ws_strategy="subdiff", fit_intercept=True, warm_start=False,
                 verbose=0, inner="squarem"):
        self.inner = inner
        self.n_pn = 0
        self.n_epochs = 0
        self.p0 = p0
        self.max_iter = max_iter
        self.max_pn_iter = max_pn_iter
        self.tol = tol
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def _solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        if self.ws_strategy not in ("subdiff", "fixpoint"):
            raise ValueError("ws_strategy must be `subdiff` or `fixpoint`, "
                             f"got {self.ws_strategy}.")
        dtype = X.dtype
        n_samples, n_features = X.shape
        fit_intercept = self.fit_intercept

        w = np.zeros(n_features + fit_intercept, dtype) if w_init is None else w_init
        Xw = np.zeros(n_samples, dtype) if Xw_init is None else Xw_init
        all_features = np.arange(n_features)
        stop_crit = 0.
        p_objs_out = []

        is_sparse = issparse(X)
        if is_sparse:
            X_bundles = (X.data, X.indptr, X.indices)

        # TODO: to be isolated in a seperated method
        if is_sparse:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
        else:
            datafit.initialize(X, y)

        if self.ws_strategy == "fixpoint":
            X_square = X.multiply(X) if is_sparse else X ** 2

        if len(w) != n_features + self.fit_intercept:
            if self.fit_intercept:
                val_error_message = (
                    "w should be of size n_features + 1 when using fit_intercept=True: "
                    f"expected {n_features + 1}, got {len(w)}.")
            else:
                val_error_message = (
                    "w should be of size n_features: "
                    f"expected {n_features}, got {len(w)}.")
            raise ValueError(val_error_message)

        self.n_pn = 0
        self.n_epochs = 0
        sq_state = np.array([1.0, 0.0])   # a_max, cycle counter, carried across inner solves
        if self.inner == "squarem" and self.ws_strategy != "subdiff":
            raise ValueError("inner='squarem': ws_strategy='subdiff' only")
        for t in range(self.max_iter):
            # compute scores
            if is_sparse:
                grad = _construct_grad_sparse(
                    *X_bundles, y, w[:n_features], Xw, datafit, all_features)
            else:
                grad = _construct_grad(X, y, w[:n_features], Xw, datafit, all_features)

            if self.ws_strategy == "subdiff":
                opt = penalty.subdiff_distance(w[:n_features], grad, all_features)
            elif self.ws_strategy == "fixpoint":
                lipschitz = datafit.raw_hessian(y, Xw) @ X_square
                opt = dist_fix_point_cd(
                    w[:n_features], grad, lipschitz, datafit, penalty, all_features
                )

            # optimality of intercept
            if fit_intercept:
                # gradient w.r.t. intercept (constant features of ones)
                intercept_opt = np.abs(np.sum(datafit.raw_grad(y, Xw)))
            else:
                intercept_opt = 0.

            # check convergences
            stop_crit = max(np.max(opt), intercept_opt)
            if self.verbose:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w[:n_features])
                print(
                    "Iteration {}: {:.10f}, ".format(t+1, p_obj) +
                    "stopping crit: {:.2e}".format(stop_crit)
                )

            if stop_crit <= self.tol:
                if self.verbose:
                    print("Stopping criterion max violation: {:.2e}".format(stop_crit))
                break

            # build working set
            gsupp_size = penalty.generalized_support(w).sum()
            ws_size = max(min(self.p0, n_features),
                          min(n_features, 2 * gsupp_size))
            # similar to np.argsort()[-ws_size:] but without sorting
            ws = np.argpartition(opt, -ws_size)[-ws_size:]

            grad_ws = grad[ws]
            tol_in = EPS_TOL * stop_crit

            for pn_iter in range(self.max_pn_iter):
                # find descent direction
                if is_sparse:
                    self.n_pn += 1
                    if self.inner == "squarem":
                        delta_w_ws, X_delta_w_ws, lipschitz_ws, ne = _descent_direction_sq_s(
                            *X_bundles, y, w, Xw, fit_intercept, grad_ws, datafit, penalty, ws, EPS_TOL*tol_in, sq_state)
                    else:
                        delta_w_ws, X_delta_w_ws, lipschitz_ws, ne = _descent_direction_s(
                            *X_bundles, y, w, Xw, fit_intercept, grad_ws, datafit,
                            penalty, ws, tol=EPS_TOL*tol_in, ws_strategy=self.ws_strategy)
                    self.n_epochs += ne
                else:
                    self.n_pn += 1
                    if self.inner == "squarem":
                        delta_w_ws, X_delta_w_ws, lipschitz_ws, ne = _descent_direction_sq(
                            X, y, w, Xw, fit_intercept, grad_ws, datafit, penalty, ws, EPS_TOL*tol_in, sq_state)
                    else:
                        delta_w_ws, X_delta_w_ws, lipschitz_ws, ne = _descent_direction(
                            X, y, w, Xw, fit_intercept, grad_ws, datafit,
                            penalty, ws, tol=EPS_TOL*tol_in, ws_strategy=self.ws_strategy)
                    self.n_epochs += ne

                # backtracking line search with inplace update of w, Xw
                if is_sparse:
                    grad_ws[:] = _backtrack_line_search_s(
                        *X_bundles, y, w, Xw, fit_intercept, datafit, penalty,
                        delta_w_ws, X_delta_w_ws, ws)
                else:
                    grad_ws[:] = _backtrack_line_search(
                        X, y, w, Xw, fit_intercept, datafit, penalty,
                        delta_w_ws, X_delta_w_ws, ws)

                # check convergence
                if self.ws_strategy == "subdiff":
                    opt_in = penalty.subdiff_distance(w, grad_ws, ws)
                elif self.ws_strategy == "fixpoint":
                    opt_in = dist_fix_point_cd(
                        w, grad_ws, lipschitz_ws, datafit, penalty, ws
                    )
                stop_crit_in = np.max(opt_in)

                if max(self.verbose-1, 0):
                    p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                    print(
                        "PN iteration {}: {:.10f}, ".format(pn_iter+1, p_obj) +
                        "stopping crit in: {:.2e}".format(stop_crit_in)
                    )

                if stop_crit_in <= tol_in:
                    if max(self.verbose-1, 0):
                        print("Early exit")
                    break

            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            p_objs_out.append(p_obj)
        else:
            warnings.warn(
                f"`ProxNewton` did not converge for tol={self.tol:.3e} "
                f"and max_iter={self.max_iter}.\n"
                "Consider increasing `max_iter` and/or `tol`.",
                category=ConvergenceWarning
            )
        return w, np.asarray(p_objs_out), stop_crit

    def custom_checks(self, X, y, datafit, penalty):
        # ws strategy
        if self.ws_strategy == "subdiff" and not hasattr(penalty, "subdiff_distance"):
            raise AttributeError(
                "Penalty must implement `subdiff_distance` "
                "to use ws_strategy='subdiff' in ProxNewton solver"
            )


@njit
def _descent_direction(X, y, w_epoch, Xw_epoch, fit_intercept, grad_ws, datafit,
                       penalty, ws, tol, ws_strategy):
    # Given:
    #   1) b = \nabla F(X w_epoch)
    #   2) D = \nabla^2 F(X w_epoch)  <------>  raw_hess
    # Minimize quadratic approximation for delta_w = w - w_epoch:
    #  b.T @ X @ delta_w + \
    #  1/2 * delta_w.T @ (X.T @ D @ X) @ delta_w + penalty(w)
    dtype = X.dtype
    raw_hess = datafit.raw_hessian(y, Xw_epoch)

    lipschitz_ws = np.zeros(len(ws), dtype)
    for idx, j in enumerate(ws):
        lipschitz_ws[idx] = raw_hess @ X[:, j] ** 2

    # for a less costly stopping criterion, we do not compute the exact gradient,
    # but store each coordinate-wise gradient every time we update one coordinate
    past_grads = np.zeros(len(ws), dtype)
    X_delta_w_ws = np.zeros(X.shape[0], dtype)
    ws_intercept = np.append(ws, -1) if fit_intercept else ws
    w_ws = w_epoch[ws_intercept]

    if fit_intercept:
        lipschitz_intercept = np.sum(raw_hess)
        grad_intercept = np.sum(datafit.raw_grad(y, Xw_epoch))

    n_ep = 0
    for cd_iter in range(MAX_CD_ITER):
        n_ep += 1
        for idx, j in enumerate(ws):
            # skip when X[:, j] == 0
            if lipschitz_ws[idx] == 0:
                continue

            past_grads[idx] = grad_ws[idx] + X[:, j] @ (raw_hess * X_delta_w_ws)
            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz_ws[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * past_grads[idx], stepsize, j)

            if w_ws[idx] != old_w_idx:
                X_delta_w_ws += (w_ws[idx] - old_w_idx) * X[:, j]

        if fit_intercept:
            past_grads_intercept = grad_intercept + raw_hess @ X_delta_w_ws
            old_intercept = w_ws[-1]
            w_ws[-1] -= past_grads_intercept / lipschitz_intercept

            if w_ws[-1] != old_intercept:
                X_delta_w_ws += w_ws[-1] - old_intercept

        if cd_iter % 5 == 0:
            # TODO: can be improved by passing in w_ws but breaks for WeightedL1
            current_w = w_epoch.copy()
            current_w[ws_intercept] = w_ws
            if ws_strategy == "subdiff":
                opt = penalty.subdiff_distance(current_w, past_grads, ws)
            elif ws_strategy == "fixpoint":
                opt = dist_fix_point_cd(
                    current_w, past_grads, lipschitz_ws, datafit, penalty, ws
                )
            stop_crit = np.max(opt)

            if fit_intercept:
                stop_crit = max(stop_crit, np.abs(past_grads_intercept))

            if stop_crit <= tol:
                break

    # descent direction
    return w_ws - w_epoch[ws_intercept], X_delta_w_ws, lipschitz_ws, n_ep


# sparse version of _descent_direction
@njit
def _descent_direction_s(X_data, X_indptr, X_indices, y, w_epoch,
                         Xw_epoch, fit_intercept, grad_ws, datafit, penalty, ws, tol,
                         ws_strategy):
    dtype = X_data.dtype
    raw_hess = datafit.raw_hessian(y, Xw_epoch)

    lipschitz_ws = np.zeros(len(ws), dtype)
    for idx, j in enumerate(ws):
        # equivalent to: lipschitz_ws[idx] += raw_hess * X[:, j] ** 2
        lipschitz_ws[idx] = _sparse_squared_weighted_norm(
            X_data, X_indptr, X_indices, j, raw_hess)

    # see _descent_direction() comment
    past_grads = np.zeros(len(ws), dtype)
    X_delta_w_ws = np.zeros(Xw_epoch.shape[0], dtype)
    ws_intercept = np.append(ws, -1) if fit_intercept else ws
    w_ws = w_epoch[ws_intercept]

    if fit_intercept:
        lipschitz_intercept = np.sum(raw_hess)
        grad_intercept = np.sum(datafit.raw_grad(y, Xw_epoch))

    n_ep = 0
    for cd_iter in range(MAX_CD_ITER):
        n_ep += 1
        for idx, j in enumerate(ws):
            # skip when X[:, j] == 0
            if lipschitz_ws[idx] == 0:
                continue

            past_grads[idx] = grad_ws[idx]
            # equivalent to cached_grads[idx] += X[:, j] @ (raw_hess * X_delta_w_ws)
            past_grads[idx] += _sparse_weighted_dot(
                X_data, X_indptr, X_indices, j, X_delta_w_ws, raw_hess)

            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz_ws[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * past_grads[idx], stepsize, j)

            if w_ws[idx] != old_w_idx:
                _update_X_delta_w(X_data, X_indptr, X_indices, X_delta_w_ws,
                                  w_ws[idx] - old_w_idx, j)

        if fit_intercept:
            past_grads_intercept = grad_intercept + raw_hess @ X_delta_w_ws
            old_intercept = w_ws[-1]
            w_ws[-1] -= past_grads_intercept / lipschitz_intercept

            if w_ws[-1] != old_intercept:
                X_delta_w_ws += w_ws[-1] - old_intercept

        if cd_iter % 5 == 0:
            # TODO: could be improved by passing in w_ws
            current_w = w_epoch.copy()
            current_w[ws_intercept] = w_ws
            if ws_strategy == "subdiff":
                opt = penalty.subdiff_distance(current_w, past_grads, ws)
            elif ws_strategy == "fixpoint":
                opt = dist_fix_point_cd(
                    current_w, past_grads, lipschitz_ws, datafit, penalty, ws
                )
            stop_crit = np.max(opt)

            if fit_intercept:
                stop_crit = max(stop_crit, np.abs(past_grads_intercept))

            if stop_crit <= tol:
                break

    # descent direction
    return w_ws - w_epoch[ws_intercept], X_delta_w_ws, lipschitz_ws, n_ep


@njit
def _backtrack_line_search(X, y, w, Xw, fit_intercept, datafit, penalty, delta_w_ws,
                           X_delta_w_ws, ws):
    # 1) find step in [0, 1] such that:
    #   penalty(w + step * delta_w) - penalty(w) +
    #   step * \nabla datafit(w + step * delta_w) @ delta_w < 0
    # ref: https://www.di.ens.fr/~aspremon/PDF/ENSAE/Newton.pdf
    # 2) inplace update of w and Xw and return grad_ws of the last w and Xw
    step, prev_step = 1., 0.
    n_features = X.shape[1]
    ws_intercept = np.append(ws, -1) if fit_intercept else ws
    # TODO: could be improved by passing in w[ws]
    old_penalty_val = penalty.value(w[:n_features])

    # try step = 1, 1/2, 1/4, ...
    for _ in range(MAX_BACKTRACK_ITER):
        w[ws_intercept] += (step - prev_step) * delta_w_ws
        Xw += (step - prev_step) * X_delta_w_ws

        grad_ws = _construct_grad(X, y, w[:n_features], Xw, datafit, ws)
        # TODO: could be improved by passing in w[ws]
        stop_crit = penalty.value(w[:n_features]) - old_penalty_val

        # it is mandatory to split the two operations, otherwise numba raises an error
        # cf. https://github.com/numba/numba/issues/9025
        dot = grad_ws @ delta_w_ws[:len(ws)]
        stop_crit += step * dot

        if fit_intercept:
            stop_crit += step * delta_w_ws[-1] * np.sum(datafit.raw_grad(y, Xw))

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2
    else:
        pass
        # TODO this case is not handled yet

    return grad_ws


# sparse version of _backtrack_line_search
@njit
def _backtrack_line_search_s(X_data, X_indptr, X_indices, y, w, Xw, fit_intercept,
                             datafit, penalty, delta_w_ws, X_delta_w_ws, ws):
    step, prev_step = 1., 0.
    n_features = len(X_indptr) - 1
    ws_intercept = np.append(ws, -1) if fit_intercept else ws
    # TODO: could be improved by passing in w[ws]
    old_penalty_val = penalty.value(w[:n_features])

    for _ in range(MAX_BACKTRACK_ITER):
        w[ws_intercept] += (step - prev_step) * delta_w_ws
        Xw += (step - prev_step) * X_delta_w_ws

        grad_ws = _construct_grad_sparse(X_data, X_indptr, X_indices,
                                         y, w[:n_features], Xw, datafit, ws)
        # TODO: could be improved by passing in w[ws]
        stop_crit = penalty.value(w[:n_features]) - old_penalty_val

        # it is mandatory to split the two operations, otherwise numba raises an error
        # cf. https://github.com/numba/numba/issues/9025
        dot = grad_ws.T @ delta_w_ws[:len(ws)]
        stop_crit += step * dot

        if fit_intercept:
            stop_crit += step * delta_w_ws[-1] * np.sum(datafit.raw_grad(y, Xw))

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2
    else:
        pass  # TODO

    return grad_ws


@njit
def _construct_grad(X, y, w, Xw, datafit, ws):
    # Compute grad of datafit restricted to ws. This function avoids
    # recomputing raw_grad for every j, which is costly for logreg
    raw_grad = datafit.raw_grad(y, Xw)
    grad = np.zeros(len(ws), dtype=X.dtype)
    for idx, j in enumerate(ws):
        grad[idx] = X[:, j] @ raw_grad
    return grad


@njit
def _construct_grad_sparse(X_data, X_indptr, X_indices, y, w, Xw, datafit, ws):
    # Compute grad of datafit restricted to ws in case X sparse
    raw_grad = datafit.raw_grad(y, Xw)
    grad = np.zeros(len(ws), dtype=X_data.dtype)
    for idx, j in enumerate(ws):
        grad[idx] = _sparse_xj_dot(X_data, X_indptr, X_indices, j, raw_grad)
    return grad


@njit(fastmath=True)
def _sparse_weighted_dot(X_data, X_indptr, X_indices, j, other, weights):
    # Compute X[:, j] @ (weights * other) in case X sparse
    res = 0.
    for i in range(X_indptr[j], X_indptr[j+1]):
        res += X_data[i] * other[X_indices[i]] * weights[X_indices[i]]
    return res


@njit(fastmath=True)
def _sparse_squared_weighted_norm(X_data, X_indptr, X_indices, j, weights):
    # Compute weights @ X[:, j]**2 in case X sparse
    res = 0.
    for i in range(X_indptr[j], X_indptr[j+1]):
        res += weights[X_indices[i]] * X_data[i]**2
    return res


@njit(fastmath=True)
def _update_X_delta_w(X_data, X_indptr, X_indices, X_delta_w, diff, j):
    # Compute X_delta_w += diff * X[:, j] in case of X sparse
    for i in range(X_indptr[j], X_indptr[j+1]):
        X_delta_w[X_indices[i]] += diff * X_data[i]


@njit
def _inner_epoch(X, raw_hess, grad_ws, lipschitz_ws, w_ws, X_delta_w_ws, past_grads, penalty, ws,
                 fit_intercept, grad_intercept, lipschitz_intercept):
    # one cyclic CD epoch on the weighted lasso model, as in skglm's _descent_direction, in place; with an intercept (the
    # last entry of w_ws) it is updated after the pass, as in skglm. Returns the intercept's model gradient (0 without).
    for idx, j in enumerate(ws):
        if lipschitz_ws[idx] == 0:
            continue
        past_grads[idx] = grad_ws[idx] + X[:, j] @ (raw_hess * X_delta_w_ws)
        old_w_idx = w_ws[idx]
        stepsize = 1 / lipschitz_ws[idx]
        w_ws[idx] = penalty.prox_1d(old_w_idx - stepsize * past_grads[idx], stepsize, j)
        if w_ws[idx] != old_w_idx:
            X_delta_w_ws += (w_ws[idx] - old_w_idx) * X[:, j]
    pgi = 0.
    if fit_intercept:
        pgi = grad_intercept + raw_hess @ X_delta_w_ws
        old_intercept = w_ws[-1]
        w_ws[-1] -= pgi / lipschitz_intercept
        if w_ws[-1] != old_intercept:
            X_delta_w_ws += w_ws[-1] - old_intercept
    return pgi


@njit
def _model_value(raw_hess, grad_ws, w_ws, w_epoch_ws, X_delta_w_ws, penalty, nws, fit_intercept, grad_intercept):
    # quadratic model of the datafit plus the penalty on ws, up to a constant
    v = (grad_ws @ (w_ws[:nws] - w_epoch_ws[:nws]) + 0.5 * np.sum(raw_hess * X_delta_w_ws ** 2)
         + penalty.value(w_ws[:nws]))
    if fit_intercept:
        v += grad_intercept * (w_ws[-1] - w_epoch_ws[-1])
    return v


@njit
def _stop(w_epoch, w_ws, past_grads, penalty, ws, tol, nws, pgi):
    current_w = w_epoch.copy()
    current_w[ws] = w_ws[:nws]
    return max(np.max(penalty.subdiff_distance(current_w, past_grads, ws)), abs(pgi)) <= tol


@njit
def _descent_direction_sq(X, y, w_epoch, Xw_epoch, fit_intercept, grad_ws, datafit, penalty, ws, tol, sq_state):
    raw_hess = datafit.raw_hessian(y, Xw_epoch)
    nws = len(ws)
    lipschitz_ws = np.zeros(nws)
    for idx, j in enumerate(ws):
        lipschitz_ws[idx] = raw_hess @ X[:, j] ** 2
    grad_intercept = 0.
    lipschitz_intercept = 1.
    if fit_intercept:
        lipschitz_intercept = np.sum(raw_hess)
        grad_intercept = np.sum(datafit.raw_grad(y, Xw_epoch))
        ws_i = np.append(ws, -1)
    else:
        ws_i = ws
    w_epoch_ws = w_epoch[ws_i].copy()
    w0 = w_epoch_ws.copy()
    X0 = np.zeros(X.shape[0])
    past_grads = np.zeros(nws)
    pgi = _inner_epoch(X, raw_hess, grad_ws, lipschitz_ws, w0, X0, past_grads, penalty, ws,
                       fit_intercept, grad_intercept, lipschitz_intercept)
    n_ep = 1
    if _stop(w_epoch, w0, past_grads, penalty, ws, tol, nws, pgi):
        return w0 - w_epoch_ws, X0, lipschitz_ws, n_ep
    q0 = _model_value(raw_hess, grad_ws, w0, w_epoch_ws, X0, penalty, nws, fit_intercept, grad_intercept)
    while n_ep < MAX_CD_ITER:
        if n_ep + 3 > MAX_CD_ITER:   # too few epochs left for a cycle: plain epochs
            pgi = _inner_epoch(X, raw_hess, grad_ws, lipschitz_ws, w0, X0, past_grads, penalty, ws,
                               fit_intercept, grad_intercept, lipschitz_intercept)
            n_ep += 1
            if _stop(w_epoch, w0, past_grads, penalty, ws, tol, nws, pgi):
                break
            continue
        w1 = w0.copy(); X1 = X0.copy()
        _inner_epoch(X, raw_hess, grad_ws, lipschitz_ws, w1, X1, past_grads, penalty, ws,
                     fit_intercept, grad_intercept, lipschitz_intercept)
        w2 = w1.copy(); X2 = X1.copy()
        _inner_epoch(X, raw_hess, grad_ws, lipschitz_ws, w2, X2, past_grads, penalty, ws,
                     fit_intercept, grad_intercept, lipschitz_intercept)
        n_ep += 2
        d1 = X1 - X0
        vv = (X2 - X1) - d1
        sv2 = np.sum(raw_hess * vv * vv)
        sq_state[1] += 1.0
        a = 1.0
        if sv2 > 0:
            if sq_state[1] % 2 == 1:
                a = -np.sum(raw_hess * d1 * vv) / sv2
            else:
                den = np.sum(raw_hess * d1 * vv)
                if den < 0:
                    a = -np.sum(raw_hess * d1 * d1) / den
        if not np.isfinite(a):
            a = 1.0
        a = min(sq_state[0], max(1.0, a))
        wx = w0 + 2 * a * (w1 - w0) + a * a * ((w2 - w1) - (w1 - w0))
        if a > 10:   # rounding in the a^2 term: recompute X delta from wx (one matrix-vector product)
            Xx = np.zeros(X.shape[0])
            for idx, j in enumerate(ws):
                if wx[idx] != w_epoch_ws[idx]:
                    Xx += (wx[idx] - w_epoch_ws[idx]) * X[:, j]
            if fit_intercept:
                Xx += wx[-1] - w_epoch_ws[-1]
        else:
            Xx = X0 + 2 * a * d1 + a * a * vv
        pgi = _inner_epoch(X, raw_hess, grad_ws, lipschitz_ws, wx, Xx, past_grads, penalty, ws,
                           fit_intercept, grad_intercept, lipschitz_intercept)
        n_ep += 1
        q3 = _model_value(raw_hess, grad_ws, wx, w_epoch_ws, Xx, penalty, nws, fit_intercept, grad_intercept)
        rejected = False
        if not np.isfinite(q3) or q3 > q0:
            rejected = True
            wx = w2; Xx = X2
            q3 = _model_value(raw_hess, grad_ws, w2, w_epoch_ws, X2, penalty, nws, fit_intercept, grad_intercept)
            if a == sq_state[0]:
                sq_state[0] = max(1.0, sq_state[0] / 4)
        elif a == sq_state[0]:
            sq_state[0] = 4 * sq_state[0]
        w0 = wx; X0 = Xx; q0 = q3
        if not rejected and _stop(w_epoch, w0, past_grads, penalty, ws, tol, nws, pgi):
            break
    return w0 - w_epoch_ws, X0, lipschitz_ws, n_ep


@njit
def _inner_epoch_s(X_data, X_indptr, X_indices, raw_hess, grad_ws, lipschitz_ws, w_ws, X_delta_w_ws, past_grads, penalty, ws,
                 fit_intercept, grad_intercept, lipschitz_intercept):
    # one cyclic CD epoch on the weighted lasso model, as in skglm's _descent_direction, in place; with an intercept (the
    # last entry of w_ws) it is updated after the pass, as in skglm. Returns the intercept's model gradient (0 without).
    for idx, j in enumerate(ws):
        if lipschitz_ws[idx] == 0:
            continue
        past_grads[idx] = grad_ws[idx] + _sparse_weighted_dot(X_data, X_indptr, X_indices, j, X_delta_w_ws, raw_hess)
        old_w_idx = w_ws[idx]
        stepsize = 1 / lipschitz_ws[idx]
        w_ws[idx] = penalty.prox_1d(old_w_idx - stepsize * past_grads[idx], stepsize, j)
        if w_ws[idx] != old_w_idx:
            _update_X_delta_w(X_data, X_indptr, X_indices, X_delta_w_ws, w_ws[idx] - old_w_idx, j)
    pgi = 0.
    if fit_intercept:
        pgi = grad_intercept + raw_hess @ X_delta_w_ws
        old_intercept = w_ws[-1]
        w_ws[-1] -= pgi / lipschitz_intercept
        if w_ws[-1] != old_intercept:
            X_delta_w_ws += w_ws[-1] - old_intercept
    return pgi



@njit
def _descent_direction_sq_s(X_data, X_indptr, X_indices, y, w_epoch, Xw_epoch, fit_intercept, grad_ws, datafit, penalty, ws, tol, sq_state):
    raw_hess = datafit.raw_hessian(y, Xw_epoch)
    nws = len(ws)
    lipschitz_ws = np.zeros(nws)
    for idx, j in enumerate(ws):
        lipschitz_ws[idx] = _sparse_squared_weighted_norm(X_data, X_indptr, X_indices, j, raw_hess)
    grad_intercept = 0.
    lipschitz_intercept = 1.
    if fit_intercept:
        lipschitz_intercept = np.sum(raw_hess)
        grad_intercept = np.sum(datafit.raw_grad(y, Xw_epoch))
        ws_i = np.append(ws, -1)
    else:
        ws_i = ws
    w_epoch_ws = w_epoch[ws_i].copy()
    w0 = w_epoch_ws.copy()
    X0 = np.zeros(Xw_epoch.shape[0])
    past_grads = np.zeros(nws)
    pgi = _inner_epoch_s(X_data, X_indptr, X_indices, raw_hess, grad_ws, lipschitz_ws, w0, X0, past_grads, penalty, ws,
                       fit_intercept, grad_intercept, lipschitz_intercept)
    n_ep = 1
    if _stop(w_epoch, w0, past_grads, penalty, ws, tol, nws, pgi):
        return w0 - w_epoch_ws, X0, lipschitz_ws, n_ep
    q0 = _model_value(raw_hess, grad_ws, w0, w_epoch_ws, X0, penalty, nws, fit_intercept, grad_intercept)
    while n_ep < MAX_CD_ITER:
        if n_ep + 3 > MAX_CD_ITER:   # too few epochs left for a cycle: plain epochs
            pgi = _inner_epoch_s(X_data, X_indptr, X_indices, raw_hess, grad_ws, lipschitz_ws, w0, X0, past_grads, penalty, ws,
                               fit_intercept, grad_intercept, lipschitz_intercept)
            n_ep += 1
            if _stop(w_epoch, w0, past_grads, penalty, ws, tol, nws, pgi):
                break
            continue
        w1 = w0.copy(); X1 = X0.copy()
        _inner_epoch_s(X_data, X_indptr, X_indices, raw_hess, grad_ws, lipschitz_ws, w1, X1, past_grads, penalty, ws,
                     fit_intercept, grad_intercept, lipschitz_intercept)
        w2 = w1.copy(); X2 = X1.copy()
        _inner_epoch_s(X_data, X_indptr, X_indices, raw_hess, grad_ws, lipschitz_ws, w2, X2, past_grads, penalty, ws,
                     fit_intercept, grad_intercept, lipschitz_intercept)
        n_ep += 2
        d1 = X1 - X0
        vv = (X2 - X1) - d1
        sv2 = np.sum(raw_hess * vv * vv)
        sq_state[1] += 1.0
        a = 1.0
        if sv2 > 0:
            if sq_state[1] % 2 == 1:
                a = -np.sum(raw_hess * d1 * vv) / sv2
            else:
                den = np.sum(raw_hess * d1 * vv)
                if den < 0:
                    a = -np.sum(raw_hess * d1 * d1) / den
        if not np.isfinite(a):
            a = 1.0
        a = min(sq_state[0], max(1.0, a))
        wx = w0 + 2 * a * (w1 - w0) + a * a * ((w2 - w1) - (w1 - w0))
        if a > 10:   # rounding in the a^2 term: recompute X delta from wx (one matrix-vector product)
            Xx = np.zeros(Xw_epoch.shape[0])
            for idx, j in enumerate(ws):
                if wx[idx] != w_epoch_ws[idx]:
                    _update_X_delta_w(X_data, X_indptr, X_indices, Xx, wx[idx] - w_epoch_ws[idx], j)
            if fit_intercept:
                Xx += wx[-1] - w_epoch_ws[-1]
        else:
            Xx = X0 + 2 * a * d1 + a * a * vv
        pgi = _inner_epoch_s(X_data, X_indptr, X_indices, raw_hess, grad_ws, lipschitz_ws, wx, Xx, past_grads, penalty, ws,
                           fit_intercept, grad_intercept, lipschitz_intercept)
        n_ep += 1
        q3 = _model_value(raw_hess, grad_ws, wx, w_epoch_ws, Xx, penalty, nws, fit_intercept, grad_intercept)
        rejected = False
        if not np.isfinite(q3) or q3 > q0:
            rejected = True
            wx = w2; Xx = X2
            q3 = _model_value(raw_hess, grad_ws, w2, w_epoch_ws, X2, penalty, nws, fit_intercept, grad_intercept)
            if a == sq_state[0]:
                sq_state[0] = max(1.0, sq_state[0] / 4)
        elif a == sq_state[0]:
            sq_state[0] = 4 * sq_state[0]
        w0 = wx; X0 = Xx; q0 = q3
        if not rejected and _stop(w_epoch, w0, past_grads, penalty, ws, tol, nws, pgi):
            break
    return w0 - w_epoch_ws, X0, lipschitz_ws, n_ep
