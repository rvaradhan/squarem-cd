"""Coordinate Newton epochs for the L1-penalized logistic regression of skglm (Logistic datafit, mean scale, labels in
{-1, 1}), dense and sparse. Coordinate j takes the step soft(w_j - g_j/h_j, alpha/h_j) with the actual curvature
h_j = sum_i x_ij^2 p_i (1 - p_i) / n (the CDN step of Yuan et al., 2010), safeguarded: the change of the objective along
the coordinate is computed without cancellation, and if it is not negative up to rounding error the majorized step of
skglm (curvature lc_j = ||x_j||^2 / (4n)) is taken instead. Each function returns the number of such fall-backs."""
import math
import numpy as np
from numba import njit


@njit
def _newton_epoch(X, y01, w, Xw, lc, alpha, ws):
    # skglm's Logistic datafit: (1/n) sum log(1 + exp(-y_i Xw_i)) with y in {-1, 1}; written here with y01 = (y + 1)/2 as
    # (1/n) sum {softplus(eta_i) - y01_i eta_i}, eta = Xw; penalty alpha |w_j| with alpha = lambda/n; lc = ||x_j||^2/(4n).
    n = X.shape[0]; nfb = 0; pr = np.empty(n); eps = 2.220446049250313e-16
    for j in ws:
        if lc[j] <= 0.0: continue
        g = 0.0; h = 0.0
        for i in range(n):
            pi = 1.0 / (1.0 + math.exp(-Xw[i])); pr[i] = pi
            g += X[i, j] * (pi - y01[i]); h += X[i, j] * X[i, j] * pi * (1.0 - pi)
        g /= n; h /= n
        if h < 1e-12 * lc[j]: h = 1e-12 * lc[j]
        z = w[j] - g / h; t = alpha / h
        bn = z - t if z > t else (z + t if z < -t else 0.0); dd = bn - w[j]
        if dd != 0.0:
            D = alpha * (abs(bn) - abs(w[j])); Asum = alpha * (abs(bn) + abs(w[j]))
            for i in range(n):
                di = dd * X[i, j]; v = y01[i] * di
                if abs(di) <= 1.0: u = math.log1p(pr[i] * math.expm1(di))
                else:
                    e0 = Xw[i]; e1 = e0 + di
                    sp1 = e1 + math.log1p(math.exp(-e1)) if e1 > 0 else math.log1p(math.exp(e1))
                    sp0 = e0 + math.log1p(math.exp(-e0)) if e0 > 0 else math.log1p(math.exp(e0))
                    u = sp1 - sp0
                D += (u - v) / n; Asum += (abs(u) + abs(v)) / n
            if not (D <= 64.0 * eps * Asum):   # no decrease (or not finite): the majorized step of skglm
                nfb += 1; z = w[j] - g / lc[j]; t = alpha / lc[j]
                bn = z - t if z > t else (z + t if z < -t else 0.0); dd = bn - w[j]
        if dd != 0.0:
            for i in range(n): Xw[i] += dd * X[i, j]
            w[j] = bn
    return nfb
@njit
def _newton_epoch_s(data, indptr, indices, y01, w, Xw, lc, alpha, ws, n):
    # sparse version of _newton_epoch: same step and safeguard, over the nonzeros of column j
    nfb = 0; eps = 2.220446049250313e-16
    for j in ws:
        if lc[j] <= 0.0: continue
        a0 = indptr[j]; a1 = indptr[j + 1]; m = a1 - a0; pr = np.empty(m)
        g = 0.0; h = 0.0
        for k in range(a0, a1):
            i = indices[k]; xij = data[k]; pi = 1.0 / (1.0 + math.exp(-Xw[i])); pr[k - a0] = pi
            g += xij * (pi - y01[i]); h += xij * xij * pi * (1.0 - pi)
        g /= n; h /= n
        if h < 1e-12 * lc[j]: h = 1e-12 * lc[j]
        z = w[j] - g / h; t = alpha / h
        bn = z - t if z > t else (z + t if z < -t else 0.0); dd = bn - w[j]
        if dd != 0.0:
            D = alpha * (abs(bn) - abs(w[j])); Asum = alpha * (abs(bn) + abs(w[j]))
            for k in range(a0, a1):
                i = indices[k]; di = dd * data[k]; v = y01[i] * di
                if abs(di) <= 1.0: u = math.log1p(pr[k - a0] * math.expm1(di))
                else:
                    e0 = Xw[i]; e1 = e0 + di
                    sp1 = e1 + math.log1p(math.exp(-e1)) if e1 > 0 else math.log1p(math.exp(e1))
                    sp0 = e0 + math.log1p(math.exp(-e0)) if e0 > 0 else math.log1p(math.exp(e0))
                    u = sp1 - sp0
                D += (u - v) / n; Asum += (abs(u) + abs(v)) / n
            if not (D <= 64.0 * eps * Asum):
                nfb += 1; z = w[j] - g / lc[j]; t = alpha / lc[j]
                bn = z - t if z > t else (z + t if z < -t else 0.0); dd = bn - w[j]
        if dd != 0.0:
            for k in range(a0, a1): Xw[indices[k]] += dd * data[k]
            w[j] = bn
    return nfb
