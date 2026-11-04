"""Powell's conjugate-direction method (scipy.optimize.minimize), derivative-free,
in the unit cube. Restart-only: see _local.py.

args:
  x0     start in real coordinates (first block; later blocks start at the best point).
  xtol, ftol   scipy options (defaults 1e-4, 1e-4).
"""
from scipy.optimize import minimize

from _local import Local


class Optimizer(Local):
    def __init__(self, space, seed=0, x0=None, xtol=1e-4, ftol=1e-4):
        super().__init__(space, seed, x0)
        self.options = {"xtol": xtol, "ftol": ftol, "maxfev": 10 ** 9, "maxiter": 10 ** 9}

    def solve(self, f, x0):
        minimize(f, x0, method="Powell", bounds=[(0, 1)] * self.dim, options=self.options)
