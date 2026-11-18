"""Compare squarem-cd with skglm on a simulated problem where coordinate descent is slow (strongly correlated columns).

Usage: python compare_with_skglm.py
"""
import time

import numpy as np
from skglm import Lasso as SkLasso, SparseLogisticRegression as SkSLR
from skglm.utils.data import make_correlated_data

from squarem_cd import Lasso, SparseLogisticRegression


def timed_fit(est, X, y, repeats=3):
    est.fit(X, y)   # compile the numba code first
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); est.fit(X, y); times.append(time.perf_counter() - t0)
    return np.median(times), est


X, y, _ = make_correlated_data(n_samples=500, n_features=5000, rho=0.95, density=0.02, random_state=0)
alpha = 0.01 * np.max(np.abs(X.T @ y)) / len(y)
print("Lasso, n = 500, p = 5,000, neighbouring columns correlated 0.95, alpha = 0.01 alpha_max, tol = 1e-8")
for name, est in [("skglm.Lasso (Anderson CD)", SkLasso(alpha=alpha, tol=1e-8, max_iter=200)),
                  ("squarem_cd.Lasso (SQUAREM CD)", Lasso(alpha=alpha, tol=1e-8, max_iter=200))]:
    t, e = timed_fit(est, X, y)
    print(f"  {name:32s} {t:6.3f} s, {np.count_nonzero(e.coef_)} nonzero coefficients")

yb = np.sign(X @ np.where(np.arange(5000) % 50 == 0, 1.0, 0.0) + 0.5 * np.random.default_rng(0).standard_normal(500))
alpha = 0.01 * np.max(np.abs(X.T @ yb)) / (2 * len(yb))
print("Sparse logistic regression, same X, alpha = 0.01 alpha_max, tol = 1e-8")
for name, est in [("skglm (proximal Newton)", SkSLR(alpha=alpha, tol=1e-8, max_iter=200)),
                  ("squarem_cd (proximal Newton + SQUAREM)", SparseLogisticRegression(alpha=alpha, tol=1e-8, max_iter=200))]:
    t, e = timed_fit(est, X, yb)
    print(f"  {name:40s} {t:6.3f} s, {np.count_nonzero(e.coef_)} nonzero coefficients")
