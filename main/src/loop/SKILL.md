# Optimization skill (placeholder; written in phase 3)

Read PROBLEM.md, space.json, data.csv, decisions.md and agent/context.md.
Analyze the data with scripts in work/. Then write:

- agent/proposal.json: {"script": "lib/<name>.py" or "work/<name>.py", "label": "<name>",
  "args": {...}, "n_points": K, "state": "continue" or "restart", "warm_start": false}
  K = 0 stops the campaign.
- agent/decision.md: the decision and its reasons.
- agent/context.md: notes for your next call.
