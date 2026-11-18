"""Drop-in versions of skglm's Lasso and SparseLogisticRegression that use the SQUAREM-accelerated solvers.

Lasso takes the same arguments as skglm.Lasso and uses SquaremCD in place of skglm's AndersonCD.
SparseLogisticRegression takes the same arguments as skglm.SparseLogisticRegression and, by default, uses
ProxNewtonSquarem (skglm's default solver, ProxNewton, with SQUAREM-H in the inner solve); with solver="cd" it uses
SquaremCD with the coordinate Newton step instead.
"""
from skglm.datafits import Logistic, Quadratic
from skglm.estimators import Lasso as _Lasso, SparseLogisticRegression as _SLR, _glm_fit
from skglm.penalties import L1, L1_plus_L2

from .solver import SquaremCD
from .prox_newton import ProxNewtonSquarem


class Lasso(_Lasso):
    """skglm's Lasso, solved by coordinate descent with working sets and SQUAREM (SquaremCD).

    Objective: (1 / (2 n_samples)) ||y - Xw||^2 + alpha ||w||_1. Parameters as in skglm.Lasso.
    """

    def fit(self, X, y):
        solver = SquaremCD(self.max_iter, self.max_epochs, self.p0, tol=self.tol, ws_strategy=self.ws_strategy,
                           fit_intercept=self.fit_intercept, warm_start=self.warm_start, verbose=self.verbose)
        self.solver_ = solver
        return _glm_fit(X, y, self, Quadratic(), L1(self.alpha, self.positive), solver)


class SparseLogisticRegression(_SLR):
    """skglm's SparseLogisticRegression with SQUAREM acceleration.

    Objective: (1 / n_samples) sum log(1 + exp(-y_i x_i'w)) + alpha (l1_ratio ||w||_1 + (1 - l1_ratio)/2 ||w||^2).
    Parameters as in skglm.SparseLogisticRegression, plus

    solver : {"prox_newton", "cd"}, default "prox_newton"
        "prox_newton": ProxNewtonSquarem (proximal Newton with SQUAREM-H in the inner solve), the fastest in our tests;
        "cd": SquaremCD with the coordinate Newton step (L1 penalty only, l1_ratio = 1).
    """

    def __init__(self, alpha=1.0, l1_ratio=1.0, tol=1e-4, max_iter=20, max_epochs=1_000, verbose=0, fit_intercept=True,
                 warm_start=False, solver="prox_newton"):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter, max_epochs=max_epochs,
                         verbose=verbose, fit_intercept=fit_intercept, warm_start=warm_start)
        self.solver = solver

    def fit(self, X, y):
        if self.solver == "prox_newton":
            solver = ProxNewtonSquarem(max_iter=self.max_iter, max_pn_iter=self.max_epochs, tol=self.tol,
                                       fit_intercept=self.fit_intercept, warm_start=self.warm_start, verbose=self.verbose)
            penalty = L1_plus_L2(self.alpha, self.l1_ratio)
        elif self.solver == "cd":
            if self.l1_ratio != 1.0:
                raise ValueError('solver="cd" supports only l1_ratio=1 (the L1 penalty).')
            solver = SquaremCD(max_iter=max(self.max_iter, 50), tol=self.tol, fit_intercept=self.fit_intercept,
                               warm_start=self.warm_start, newton_step=True, verbose=self.verbose)
            penalty = L1(self.alpha)
        else:
            raise ValueError(f'solver must be "prox_newton" or "cd", got {self.solver!r}.')
        self.solver_ = solver
        return _glm_fit(X, y, self, Logistic(), penalty, solver)
