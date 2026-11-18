# optimization

Portfolio mathematics. Every function here works at a single rebalancing date and takes plain
pandas inputs; nothing knows about the calendar or the backtest loop.

```
returns up to t → Ledoit–Wolf covariance Σ ─┐
                                            ├→ equilibrium prior π = δ Σ w_ref
reference weights w_ref ────────────────────┘
views (P, q, Ω) from simulation.views ──→ Black–Litterman posterior (μ_BL, Σ_BL) ──→ QP ──→ target weights
```

| File | Purpose |
|---|---|
| [ledoit_wolf.py](ledoit_wolf.py) | Ledoit–Wolf (2004) shrinkage covariance over a trailing window, scaled to the 4-week horizon. |
| [equilibrium_prior.py](equilibrium_prior.py) | Reference portfolio `w_ref` from the config and implied equilibrium excess returns by reverse optimization. |
| [black_litterman.py](black_litterman.py) | Posterior mean and covariance given views, in Woodbury form. |
| [optimizer.py](optimizer.py) | Long-only mean–variance QP with a weight cap and an L1 turnover penalty (CVXPY + Clarabel), a closed-form solution for testing, and a KKT optimality check. |
| [baselines.py](baselines.py) | Comparator allocations: equal weight, inverse volatility, minimum variance, Markowitz with sample means, and Black–Litterman with no views. |

## Parameters

Set in [config.yaml](../config.yaml): `risk` (covariance window and horizon), `prior`
(`delta`, `tau`), `optimizer` (`gamma`, `w_max`, `turnover_penalty`, `solver`). Override per
experiment with `load_config(overrides=...)`.

## Tests

Closed-form and KKT checks keep the solver honest:
[test_ledoit_wolf.py](../tests/test_ledoit_wolf.py),
[test_prior.py](../tests/test_prior.py),
[test_black_litterman.py](../tests/test_black_litterman.py),
[test_optimizer.py](../tests/test_optimizer.py),
[test_baselines.py](../tests/test_baselines.py).
