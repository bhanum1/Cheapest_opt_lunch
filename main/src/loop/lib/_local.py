"""Restart-only base for local solvers whose libraries call f themselves.

The solver runs in a thread inside one block. Its internal state (model,
line search, trust radius) cannot be saved, so saved state holds only the
settings and the data. Every block, including `continue`, starts the solver
afresh from x0 on the first block, then from the best point this optimizer
has been told. When the solver converges, the block ends early.
"""
import queue
import threading

import numpy as np

from _base import Base


class Local(Base):
    def __init__(self, space, seed=0, x0=None):
        super().__init__(space, seed)
        if self.indexed:
            raise ValueError("local solvers need a continuous space, not an indexed pool")
        self.x0 = None if x0 is None else self.to_unit(x0)
        self.thread, self.starts = None, 0
        self.last = None  # raw (y, g) of the latest observation, for constrained solvers

    def solve(self, f, x0):
        """Run the solver on f (unit-cube point -> value) from x0."""
        raise NotImplementedError

    def propose(self):
        if self.thread is None:
            x0 = self.x0 if self.x0 is not None and not self.starts else (
                self.best()[0] if len(self.y) else np.full(self.dim, 0.5))
            self.starts += 1
            self.asks, self.answers = queue.Queue(), queue.Queue()

            def f(u):
                self.asks.put(np.clip(np.asarray(u, dtype=float), 0, 1))
                return self.answers.get()

            def run():
                try:
                    self.solve(f, np.clip(x0, 0, 1))
                    self.asks.put(None)
                except BaseException as e:  # surface solver errors in the host
                    self.asks.put(e)

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()
        item = self.asks.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def tell(self, X, y, g):
        for x, yi, gi in zip(X, y, g):
            self.last = (float(yi), [float(v) for v in gi])
            super().tell([x], [yi], [gi])

    def observe(self, u, value):
        self.answers.put(value)

    def __getstate__(self):
        state = dict(self.__dict__, thread=None)
        state.pop("asks", None)
        state.pop("answers", None)
        state["pending"] = {}  # an asked, untold point belongs to the stopped solver
        return state
