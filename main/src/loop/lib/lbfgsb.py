"""L-BFGS-B (scipy.optimize.minimize) with forward finite-difference
gradients, in the unit cube. Restart-only: see _local.py.

args:
  x0     start in real coordinates (first block; later blocks start at the best point).
  eps    finite-difference step in the unit cube (default 1e-8, scipy's default).
  ftol, gtol, maxcor   scipy options (defaults 2.2e-9, 1e-5, 10).
Each gradient costs dim + 1 evaluations. Unsuited to noisy objectives.
"""
from scipy.optimize import minimize

from _local import Local


class Optimizer(Local):
    def __init__(self, space, seed=0, x0=None, eps=1e-8, ftol=2.2e-9, gtol=1e-5, maxcor=10):
        super().__init__(space, seed, x0)
        self.options = {"eps": eps, "ftol": ftol, "gtol": gtol, "maxcor": maxcor,
                        "maxfun": 10 ** 9, "maxiter": 10 ** 9}

    def solve(self, f, x0):
        minimize(f, x0, method="L-BFGS-B", bounds=[(0, 1)] * self.dim, options=self.options)
