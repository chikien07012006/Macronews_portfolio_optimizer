# MacroNews Portfolio Lab

A news- and momentum-driven portfolio backtesting simulator for U.S. sector ETFs
(COMP3090 Course-related Project 2, VinUniversity).

Federal Reserve communications and financial news are scored by a multi-task FinBERT
model (macro topic, sentiment, hawkish/dovish stance) and aggregated into weekly news
indices. These produce *news views*; momentum, short-term reversal and volatility produce
*technical views*. Black–Litterman blends both with market-implied returns, and a long-only
mean-variance QP with a turnover penalty rebalances the portfolio weekly inside a
walk-forward backtest.

> Educational research tool only. It replays historical data, performs no trading, and is
> not investment advice.

See [docs/project_proposal.pdf](docs/project_proposal.pdf) for the full methodology and
[plan.md](plan.md) for the implementation plan and schedule.

## Status

Early scaffolding: the package skeleton, dependencies and configuration template are in
place; no pipeline, engine or backtest code exists yet. Work follows the clusters in
[plan.md](plan.md), starting with the price path (C0–C4: market data → technical views →
Black–Litterman → QP → backtest) before the text/NLP path.

## Repository layout

```
data/raw/        downloaded source data (contents git-ignored)
data/processed/  derived datasets (contents git-ignored)
docs/            project description and proposal (PDF)
src/macronews/   Python package
pyproject.toml   package metadata, pytest and ruff settings
requirements*.txt  pinned dependency sets (core / NLP)
```

Further directories (`configs/`, `scripts/`, `tests/`, `notebooks/`) are added as the
corresponding clusters are built.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # core: data, portfolio engine, backtest
pip install -e . --no-deps               # install the macronews package in editable mode
cp .env.example .env                     # then fill in FRED_API_KEY (MySQL is optional for now)
```

The NLP stack (PyTorch, Transformers) is heavy and only needed where FinBERT is trained or
used for scoring (e.g. a cloud GPU):

```bash
pip install -r requirements-nlp.txt
```

If the project folder is moved or renamed, recreate `.venv`: a virtualenv hard-codes its
absolute path.
