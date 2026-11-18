# squarem-cd

SQUAREM acceleration of coordinate descent for sparse regression, as solvers for
[skglm](https://github.com/scikit-learn-contrib/skglm).

Coordinate descent with working sets is the base of the fastest solvers for the lasso and sparse logistic regression,
but it can be slow when the active columns of the design are strongly correlated. SQUAREM (Varadhan and Roland, 2008)
is a simple, general-purpose accelerator of fixed-point iterations: it takes two steps of the map, computes a step
length in closed form from their differences, extrapolates, and takes one more step. It keeps no history and needs no
tuning. This package wraps SQUAREM around the coordinate descent epoch of skglm, with the step lengths measured in the
metric of the problem's curvature at no extra cost.

On problems where coordinate descent is slow, it reaches high accuracy two to seven times faster than skglm's own
coordinate descent solver, and with SQUAREM in its inner solve, skglm's proximal Newton solver for logistic regression is
up to three times faster. Details and benchmarks are in the paper (see Citing).

## Installation

```
pip install git+https://github.com/rvaradhan/squarem-cd
```

It requires Python 3.9 or later and skglm 0.5.

## Quick start

The estimators take the same arguments as skglm's:

```python
from squarem_cd import Lasso, SparseLogisticRegression

lasso = Lasso(alpha=0.01, tol=1e-8).fit(X, y)                    # in place of skglm.Lasso
clf = SparseLogisticRegression(alpha=0.01, tol=1e-8).fit(X, y)   # in place of skglm.SparseLogisticRegression
```

The solvers can be used with any skglm datafit and penalty that skglm's coordinate descent solver supports:

```python
from skglm import GeneralizedLinearEstimator
from skglm.datafits import Logistic
from skglm.penalties import L1
from squarem_cd import SquaremCD, ProxNewtonSquarem

est = GeneralizedLinearEstimator(Logistic(), L1(alpha), SquaremCD(tol=1e-8, newton_step=True)).fit(X, y)
est = GeneralizedLinearEstimator(Logistic(), L1(alpha), ProxNewtonSquarem(tol=1e-8)).fit(X, y)
```

`X` can be a dense array or a `scipy.sparse` matrix. See `examples/compare_with_skglm.py` for a timing comparison.

## What is in the package

| | |
|---|---|
| `SquaremCD` | Coordinate descent with working sets, accelerated by SQUAREM; a drop-in for skglm's `AndersonCD`. Options: `step` (`"alternate"`: SqS1 and SqS2 in alternation, the default), `metric` (`"H"` or `"euclidean"`), `stabilize`, `newton_step` (coordinate Newton steps for L1-penalized logistic regression), `fit_intercept`. |
| `ProxNewtonSquarem` | skglm's proximal Newton solver with SQUAREM in the inner solve of each Newton step (`inner="squarem"`; `inner="cd"` gives skglm's solver). |
| `Lasso`, `SparseLogisticRegression` | skglm's estimators with these solvers. |
| `squarem` | A generic SQUAREM accelerator for any fixed-point map, in plain NumPy. |

## When does it help?

Acceleration pays when plain coordinate descent converges slowly, typically when the active columns are strongly
correlated (gene expression, many correlated simulations, images). A simple guide from the paper: if plain coordinate
descent needs more than about 100 epochs to reduce the objective gap ten-fold, expect a large gain; with nearly
uncorrelated columns, as in many sparse text data sets, coordinate descent is already fast and the gain is small.

## Beyond regression

`squarem` accelerates any smooth, monotone fixed-point iteration, for example an EM or MM algorithm:

```python
from squarem_cd import squarem
res = squarem(em_update, x0, objfn=negative_log_likelihood)
res.x, res.n_fixptfn
```

## Citing

Varadhan, R. (2026). Accelerating coordinate descent for sparse regression with squared extrapolation (SQUAREM).
Manuscript.

Varadhan, R. and Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM
algorithm. *Scandinavian Journal of Statistics* 35, 335–353.

## License

BSD 3-Clause. `src/squarem_cd/prox_newton.py` is derived from skglm (BSD 3-Clause); see `LICENSE`.
