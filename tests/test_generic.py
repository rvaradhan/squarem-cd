import numpy as np
import pytest

from squarem_cd import squarem


def slow_linear_map(n=50, rho=0.995, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    J = Q @ np.diag(np.linspace(0, rho, n)) @ Q.T
    xstar = rng.standard_normal(n)
    b = xstar - J @ xstar
    return (lambda x: J @ x + b), xstar


@pytest.mark.parametrize("step", ["alternate", "sqs1", "sqs2", "sqs3"])
def test_linear_fixed_point(step):
    F, xstar = slow_linear_map()
    res = squarem(F, np.zeros_like(xstar), step=step, tol=1e-10, maxiter=5000)
    assert res.converged
    np.testing.assert_allclose(res.x, xstar, atol=1e-7)
    # far fewer map evaluations than the plain iteration, which needs about log(1e-10)/log(0.995) ~ 4,600
    assert res.n_fixptfn < 1500


def test_em_poisson_mixture_with_objective():
    # EM for a two-component Poisson mixture (Hasselblad's data, as in the SQUAREM package)
    y = np.repeat(np.arange(10), [162, 267, 271, 185, 111, 61, 27, 8, 3, 1]).astype(float)
    from math import lgamma
    lg = np.array([lgamma(v + 1) for v in y])

    def dens(lam):
        return np.exp(y * np.log(lam) - lam - lg)

    def em(par):
        p, l1, l2 = par
        a, b = p * dens(l1), (1 - p) * dens(l2)
        z = a / (a + b)
        return np.array([z.mean(), (z * y).sum() / z.sum(), ((1 - z) * y).sum() / (1 - z).sum()])

    def nll(par):
        p, l1, l2 = par
        return -np.sum(np.log(p * dens(l1) + (1 - p) * dens(l2)))

    x0 = np.array([0.3, 1.0, 2.5])
    res = squarem(em, x0, objfn=nll, tol=1e-9, maxiter=2000)
    plain = x0.copy()
    for _ in range(20000):
        plain = em(plain)
    assert res.converged
    np.testing.assert_allclose(res.x, plain, rtol=1e-5)
    assert res.value <= nll(plain) + 1e-8
    assert res.n_fixptfn < 500


def test_bad_step_name():
    F, xstar = slow_linear_map(n=5)
    with pytest.raises(ValueError):
        squarem(F, np.zeros(5), step="sqs4")
