"""squarem-cd: SQUAREM acceleration of coordinate descent for sparse regression, as solvers for skglm.

SquaremCD             coordinate descent with working sets, accelerated by SQUAREM (a drop-in for skglm's AndersonCD)
ProxNewtonSquarem     skglm's proximal Newton solver with SQUAREM in the inner solve of each Newton step
Lasso, SparseLogisticRegression
                      drop-in versions of skglm's estimators that use these solvers
squarem               a generic SQUAREM accelerator for any fixed-point map
"""
from .solver import SquaremCD
from .prox_newton import ProxNewtonSquarem
from .estimators import Lasso, SparseLogisticRegression
from .generic import squarem, SquaremResult

__version__ = "0.1.0"
__all__ = ["SquaremCD", "ProxNewtonSquarem", "Lasso", "SparseLogisticRegression", "squarem", "SquaremResult",
           "__version__"]
