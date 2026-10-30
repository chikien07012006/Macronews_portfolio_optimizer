# MacroNews Portfolio Lab

A news- and momentum-driven portfolio simulator for U.S. sector ETFs
(COMP3090 Course-related Project 2, VinUniversity).

Federal Reserve communications and financial news are scored by a multi-task FinBERT model
(macro topic, sentiment, hawkish/dovish stance) and aggregated into weekly news indices that
produce *news views*. Momentum, short-term reversal and volatility produce *technical views*.
Black–Litterman blends both with market-implied returns, and a long-only mean-variance
optimizer with a turnover penalty rebalances the portfolio every week.

The central research question: **do news-based views add information beyond what prices
already reflect?**

> Educational research tool. It replays historical data only, performs no live or paper
> trading, and is not investment advice.

## How it works

The project runs in two phases.

**1. Prepare (offline, once).** All history for 2009–2023 is downloaded and stored in a
single SQLite file, `data/macronews.db`: daily ETF prices, the 3-month T-bill rate and macro
series, FOMC documents and financial news. Every document is then scored by the NLP model.
Nothing in this phase makes portfolio decisions.

**2. Replay (the simulation).** The simulator walks forward one week at a time, as if living
through the period. At each weekly decision date *t* (the last trading day of the week) it
sees **only** what was knowable at that moment:

- prices up to the close of *t*;
- documents published up to 16:00 ET on *t* (later documents count towards the next week).

From that snapshot it estimates risk, builds news and technical views, computes the
Black–Litterman posterior and solves for new target weights. Trades are executed at the
close of the next trading day; between rebalances the portfolio is marked to market daily.
Then the clock advances to week *t + 1*.

```
              prepare (once)                          replay (week by week)
 ┌──────────────────────────────────┐      ┌───────────────────────────────────────────┐
 │ prices, T-bill, FOMC, news       │      │ for each week t:                          │
 │   → data_pipeline → SQLite       │ ───▶ │   snapshot of data known at t             │
 │ documents → nlp → scores, indices│      │   → views → Black–Litterman → QP          │
 └──────────────────────────────────┘      │   → trade at t+1 close → mark to market   │
                                           └───────────────────────────────────────────┘
```

Look-ahead bias is the main failure mode of a backtest, so every component that touches time
has unit tests for it. The 2018–2023 test period is evaluated only once, at the end.

## Repository layout

```
├── config.yaml          experiment parameters (universe, periods, model settings)
├── common/              shared foundation: config loader, weekly decision calendar
├── data_pipeline/       download, clean and store data in SQLite
├── nlp/                 topic labels, baselines, FinBERT fine-tuning and scoring, news indices
├── optimization/        risk model, Black–Litterman, mean-variance QP, baseline strategies
├── simulation/          week-by-week replay, view generation, rebalancing, metrics
├── backend/             FastAPI service (planned)
├── frontend/            React web interface (planned)
├── tests/               pytest suite
├── data/                local data and the SQLite database (git-ignored)
└── docs/                project description and proposal
```

Folders are plain Python packages run from the repository root; nothing needs to be
installed as a package.

## Getting started

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional: only to change data paths
pytest
```

The NLP stack (PyTorch, Transformers) is large and only needed where FinBERT is fine-tuned or
used for scoring, typically on a cloud GPU:

```bash
pip install -r requirements-nlp.txt
```

## Configuration

All experiment parameters live in [config.yaml](config.yaml), with values taken from the
proposal. Load them with `common.config.load_config()`; individual values can be overridden
for experiments without editing the file:

```python
from common.config import load_config

cfg = load_config(overrides={"optimizer.w_max": 0.25, "news.half_life_days": 7})
```

Each saved simulation run stores the full configuration it used. Data paths
(optional `MACRONEWS_DATA_DIR`, `MACRONEWS_DB_PATH`) are read from `.env`.

## Data

| Source | Content | Access |
|---|---|---|
| Yahoo Finance | Daily adjusted prices: 9 SPDR sector ETFs, TLT, GLD, SPY (XLRE, XLC optional) | `yfinance` |
| FRED | 3-month T-bill rate (DTB3), the risk-free rate | public CSV export, no key |
| Federal Reserve | FOMC statements, minutes, press-conference transcripts | scraped |
| FNSPID | Time-stamped financial news linked to S&P 500 tickers | public dataset |
| Financial PhraseBank, Shah et al. (2023) | Labelled sentiment and hawkish/dovish sentences | public datasets |

Data files and the database are not committed; they are rebuilt by the pipeline. All sources
are used under their terms of use.

## Status

Development follows the milestones in [plan.md](plan.md): the price-based path is built
first, so a working technical-only simulation exists before the NLP path is added.

| Milestone | Scope | Status |
|---|---|---|
| C0 | Configuration, decision calendar, SQLite storage | Done |
| C1 | Market data download and validation | Next |
| C2–C4 | Optimization engine, technical views, replay and metrics | Planned |
| C5–C7 | Text data, NLP models, news indices and views | Planned |
| C8–C10 | Full evaluation, web simulator with Docker, reports | Planned |

## Documentation

- [docs/project_proposal.pdf](docs/project_proposal.pdf): full methodology
- [docs/project_description.pdf](docs/project_description.pdf): project summary
- [plan.md](plan.md): implementation plan, schedule and design decisions
