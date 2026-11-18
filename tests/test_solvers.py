import numpy as np
import pytest
import scipy.sparse as sp
from skglm import GeneralizedLinearEstimator
from skglm import Lasso as SkLasso, SparseLogisticRegression as SkSLR
from skglm.datafits import Logistic, Quadratic
from skglm.penalties import L1
from skglm.solvers import AndersonCD, ProxNewton
from skglm.utils.data import make_correlated_data

from squarem_cd import Lasso, ProxNewtonSquarem, SparseLogisticRegression, SquaremCD


def lasso_data(n=60, p=200, rho=0.9, seed=0):
    X, y, _ = make_correlated_data(n, p, rho=rho, random_state=seed)
    alpha = 0.02 * np.max(np.abs(X.T @ y)) / n
    return X, y, alpha


def logit_data(n=80, p=150, rho=0.8, seed=1, sparse=False):
    X, y, _ = make_correlated_data(n, p, rho=rho, random_state=seed, X_density=0.3 if sparse else 1)
    y = np.sign(y + 0.1)
    alpha = 0.02 * np.max(np.abs(X.T @ y)) / (2 * n)
    return (sp.csc_matrix(X) if sparse else X), y, alpha


def obj_lasso(X, y, w, b, alpha):
    r = y - X @ w - b
    return r @ r / (2 * len(y)) + alpha * np.abs(w).sum()


def obj_logit(X, y, w, b, alpha):
    return np.mean(np.logaddexp(0, -y * (X @ w + b))) + alpha * np.abs(w).sum()


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("step", ["alternate", "sqs3"])
def test_lasso_matches_skglm(fit_intercept, step):
    X, y, alpha = lasso_data()
    ref = GeneralizedLinearEstimator(Quadratic(), L1(alpha), AndersonCD(tol=1e-12, max_iter=200, fit_intercept=fit_intercept)).fit(X, y)
    est = GeneralizedLinearEstimator(Quadratic(), L1(alpha), SquaremCD(tol=1e-12, max_iter=200, fit_intercept=fit_intercept, step=step)).fit(X, y)
    f_ref = obj_lasso(X, y, ref.coef_, ref.intercept_, alpha)
    f = obj_lasso(X, y, est.coef_, est.intercept_, alpha)
    assert abs(f - f_ref) <= 1e-10 * abs(f_ref)
    np.testing.assert_allclose(est.coef_, ref.coef_, atol=1e-6)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("newton_step", [False, True])
def test_logistic_matches_skglm(sparse, fit_intercept, newton_step):
    X, y, alpha = logit_data(sparse=sparse)
    ref = GeneralizedLinearEstimator(Logistic(), L1(alpha), ProxNewton(tol=1e-12, max_iter=200, fit_intercept=fit_intercept)).fit(X, y)
    solver = SquaremCD(tol=1e-11, max_iter=200, fit_intercept=fit_intercept, newton_step=newton_step)
    est = GeneralizedLinearEstimator(Logistic(), L1(alpha), solver).fit(X, y)
    f_ref = obj_logit(X, y, ref.coef_.ravel(), ref.intercept_, alpha)
    f = obj_logit(X, y, est.coef_.ravel(), est.intercept_, alpha)
    assert abs(f - f_ref) <= 1e-9 * abs(f_ref)
    assert solver.n_epochs_ > 0


def test_newton_step_requires_logistic_l1():
    X, y, alpha = lasso_data()
    with pytest.raises(ValueError):
        GeneralizedLinearEstimator(Quadratic(), L1(alpha), SquaremCD(newton_step=True)).fit(X, y)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_prox_newton_squarem(sparse, fit_intercept):
    X, y, alpha = logit_data(sparse=sparse)
    ref = GeneralizedLinearEstimator(Logistic(), L1(alpha), ProxNewton(tol=1e-10, fit_intercept=fit_intercept)).fit(X, y)
    plain = GeneralizedLinearEstimator(Logistic(), L1(alpha), ProxNewtonSquarem(tol=1e-10, fit_intercept=fit_intercept, inner="cd")).fit(X, y)
    np.testing.assert_array_equal(plain.coef_, ref.coef_)   # inner="cd" is skglm's solver, exactly
    est = GeneralizedLinearEstimator(Logistic(), L1(alpha), ProxNewtonSquarem(tol=1e-10, fit_intercept=fit_intercept)).fit(X, y)
    f_ref = obj_logit(X, y, ref.coef_.ravel(), ref.intercept_, alpha)
    f = obj_logit(X, y, est.coef_.ravel(), est.intercept_, alpha)
    assert abs(f - f_ref) <= 1e-8 * abs(f_ref)


def test_estimators():
    X, y, alpha = lasso_data()
    a = Lasso(alpha=alpha, tol=1e-10).fit(X, y)
    b = SkLasso(alpha=alpha, tol=1e-10).fit(X, y)
    assert abs(obj_lasso(X, y, a.coef_, a.intercept_, alpha) - obj_lasso(X, y, b.coef_, b.intercept_, alpha)) < 1e-8
    assert a.predict(X).shape == (X.shape[0],)
    Xl, yl, al = logit_data()
    for solver in ["prox_newton", "cd"]:
        c = SparseLogisticRegression(alpha=al, tol=1e-10, solver=solver).fit(Xl, yl)
        d = SkSLR(alpha=al, tol=1e-10).fit(Xl, yl)
        fc = obj_logit(Xl, yl, c.coef_.ravel(), c.intercept_, al)
        fd = obj_logit(Xl, yl, d.coef_.ravel(), d.intercept_, al)
        assert abs(fc - fd) <= 1e-7 * fd
        assert set(np.unique(c.predict(Xl))) <= set(np.unique(yl))


def test_warm_start_path():
    X, y, alpha_min = lasso_data()
    est = Lasso(alpha=10 * alpha_min, tol=1e-10, warm_start=True)
    for alpha in alpha_min * np.array([10.0, 3.0, 1.0]):
        est.alpha = alpha
        est.fit(X, y)
        ref = SkLasso(alpha=alpha, tol=1e-10).fit(X, y)
        assert abs(obj_lasso(X, y, est.coef_, est.intercept_, alpha) - obj_lasso(X, y, ref.coef_, ref.intercept_, alpha)) < 1e-8
