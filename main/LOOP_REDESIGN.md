# Loop redesign plan

Replace the single long agent session (`blind_runner` + `agent_harness`) with
a Python loop that calls a fresh agent once per block.

## Structure

```
run.py  (holds objective, budget, data.csv, decisions.md)
loop:
  1. skill call   agent reads data.csv / decisions.md / context.md with tools,
                  runs offline analysis scripts, may write work/<opt>.py
                  -> proposal.json, decision entry, context.md
  2. validate     script exists, args parse, 1 <= n_points <= remaining (0 = stop)
  3. record       append decision entry to decisions.md
  4. block        sandboxed optimizer: load state -> n_points x (ask -> Python
                  evaluates -> tell) -> save state; rows appended to data.csv
  5. on crash     runner entry in decisions.md: evals used, traceback tail,
                  whether saved state is usable
end: final entry, trace.csv / meta.json / scored_trace.csv, audit
```

The objective never enters the sandbox. The agent's only route to a new `y`
is a proposed block.

## Optimizer protocol

An optimizer is a Python class in its own file:

```python
class Optimizer:
    def __init__(self, space, **args): ...   # space: dim, bounds, pool candidates, constraints
    def ask(self, n) -> list[x]: ...
    def tell(self, X, y, g) -> None: ...
    def save(self, path) -> None: ...
    def load(path) -> "Optimizer": ...       # classmethod
```

A fixed `host.py` runs inside the sandbox, imports the class, and speaks JSON
lines with the runner over stdin/stdout. Library and custom optimizers use
the same host, so agent-written optimizers are only the class.
`continue` = load saved state; `restart` = construct fresh, optionally
`tell` it existing data.

## Phases

1. **Protocol and data.** `host.py`, state directory layout, `data.csv`
   columns (eval, block, optimizer, x, y, g, pool candidate; y_true kept
   private for scoring).
2. **Runner.** `run.py`: budget, loop, validation, decisions.md appends,
   crash entries, launching the skill (`claude -p` / `codex exec`) and the
   optimizer host under the existing sandbox policy.
3. **Skill.** `SKILL.md`: problem and file locations, protocol, proposal
   format, library index, offline-only analysis, what a decision entry must
   justify, what to keep in `context.md` to avoid rereading.
4. **Library.** Savable optimizers only.
   - Port native ask/tell ones: random, Sobol, CMA-ES, DE, GP-BO, TuRBO,
     pool variants.
   - General adapters: Nevergrad, Optuna, pymoo (native Nelder-Mead,
     pattern search).
   - Local model-based methods (L-BFGS-B, BOBYQA, COBYLA): search for native
     ask/tell implementations; otherwise restart-only wrappers (state = start
     point and settings; each block runs the solver with `maxfev = n_points`).
     The current thread-based NM, BOBYQA and L-BFGS-B wrappers are removed.
5. **Outputs and audit.** Keep `trace.csv` / `meta.json` compatible with the
   plot scripts. Audit: one decision entry per block, each block within its
   `n_points`, total within budget, data.csv matches the private journal.
6. **Remove the old path.** Delete `blind_runner`, `agent_harness`,
   `scripts/run_agent.py`; update root `AGENTS.md`, README and tests.
7. **Smoke runs.** Small budget on a cheap task with each agent CLI,
   including a deliberately crashing custom optimizer. Done with Claude
   (Codex deferred: not logged in on this machine).
   - Opus, high, `constrained_ellipsoid10`, budget 40: Sobol 12, then
     interpolation probes that found feasibility, then COBYLA 22. Best
     1190.8 (optimum 1072.4); 15 tool calls; audit passed.
   - Sonnet, medium, `sphere10`, budget 20, call 1 scripted to crash on its
     third ask: the runner kept the two paid points and noted the
     traceback; the agent restarted with warm-started BOBYQA. Audit passed,
     but only 2 tool calls: analysis depth depends on model and effort.
   - Both agents spent the whole remainder in the last block, justified for
     restart-only methods.
