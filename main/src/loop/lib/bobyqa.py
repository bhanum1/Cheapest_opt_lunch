"""BOBYQA (Py-BOBYQA): derivative-free trust-region method with quadratic
models, for bound-constrained problems, in the unit cube. Restart-only:
see _local.py.

args:
  x0      start in real coordinates (first block; later blocks start at the best point).
  rhobeg  initial trust radius in the unit cube (default 0.1).
  rhoend  final trust radius (default 1e-8).
  npt     interpolation points (default 2 * dim + 1; at least dim + 2).
Building the first model costs npt evaluations, so short blocks waste much of it.
"""
import numpy as np
import pybobyqa

from _local import Local


class Optimizer(Local):
    def __init__(self, space, seed=0, x0=None, rhobeg=0.1, rhoend=1e-8, npt=None):
        super().__init__(space, seed, x0)
        self.rhobeg, self.rhoend = rhobeg, rhoend
        self.npt = npt or 2 * self.dim + 1

    def solve(self, f, x0):
        # Py-BOBYQA needs x0 at least rhobeg from each bound it may move toward.
        pybobyqa.solve(f, x0, bounds=(np.zeros(self.dim), np.ones(self.dim)), npt=self.npt,
                       rhobeg=self.rhobeg, rhoend=self.rhoend, maxfun=10 ** 9,
                       seek_global_minimum=False, do_logging=False)
