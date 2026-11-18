# common

Shared foundation used by every other package: experiment configuration and the weekly
decision calendar.

| File | Purpose |
|---|---|
| [config.py](config.py) | Loads [config.yaml](../config.yaml) into typed dataclasses (`Config` and its sections) and reads machine-specific paths from the environment (`Settings`). |
| [calendar.py](calendar.py) | Single definition of "week *t*": decision dates, execution dates and the 16:00 ET document cut-off. |

## Configuration

```python
from common.config import load_config

cfg = load_config()                                   # defaults from config.yaml
cfg = load_config(overrides={"optimizer.w_max": 0.25})  # dotted-path override, file untouched
```

`Settings.from_env()` reads `MACRONEWS_DATA_DIR` and `MACRONEWS_DB_PATH` from the environment
or `.env` (see [.env.example](../.env.example)). They default to `./data` and
`./data/macronews.db`.

## Decision calendar

All components that touch time use the same definitions, so look-ahead rules live in one place:

| Function | Meaning |
|---|---|
| `decision_dates` | Last trading day of each week (or month): the day whose close is the latest usable information. |
| `execution_dates` | Trading day `lag` sessions after each decision date; target weights are traded at its close. |
| `decision_cutoffs` | Timezone-aware 16:00 ET cut-off on each decision date. |
| `assign_week` | Maps document timestamps to the first decision date whose cut-off is at or after them. |

Trading days come from a price index (for example SPY), so no exchange-calendar library is
needed. Tests: [tests/test_calendar.py](../tests/test_calendar.py),
[tests/test_config.py](../tests/test_config.py).
