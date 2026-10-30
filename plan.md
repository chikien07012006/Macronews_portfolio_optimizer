# Implementation Plan

MacroNews Portfolio Lab · last updated 2026-10-30 (week 7)

This document describes how the system in [docs/project_proposal.pdf](docs/project_proposal.pdf)
is being built: the milestones, their order, the schedule, and where the implementation
deliberately differs from the proposal.

## Deadlines

| Milestone | Date |
|---|---|
| Progress report | 13 Nov 2026 |
| Final presentation | 08 Jan 2027 |
| Final report | 15 Jan 2027 |

## Design principles

1. **Prepare once, replay week by week.** All data are downloaded and all documents scored
   ahead of time. The simulator then steps through history and, at each decision date,
   only reads data that were available at that moment (prices up to the close, documents up
   to 16:00 ET). This point-in-time access is the single place where time is enforced.
2. **Price path first.** Prices → technical views → Black–Litterman → QP → replay gives a
   complete, working system without any NLP. It is the target for the progress report.
3. **Views are source-agnostic.** The view generator takes any feature matrix, so news
   features plug into the same code as technical features.
4. **No look-ahead is an invariant**, covered by unit tests in every milestone that handles
   time.
5. **The 2018–2023 test period is used once**, for the final results. All tuning happens on
   2013–2017.
6. **Keep the stack small.** Plain Python packages run from the repository root, one SQLite
   file for storage, one YAML file for configuration.

## Architecture

| Folder | Responsibility | Milestones |
|---|---|---|
| `common/` | Config loader, weekly decision and execution calendar | C0 |
| `data_pipeline/` | Downloading, cleaning and storing all data in SQLite | C0, C1, C5 |
| `optimization/` | Ledoit–Wolf covariance, equilibrium prior, Black–Litterman, QP, baselines | C2 |
| `simulation/` | Point-in-time replay, features and Ridge views, rebalancing, metrics | C3, C4, C7, C8 |
| `nlp/` | Topic labelling, lexicon and TF-IDF baselines, FinBERT, scoring, news indices | C6, C7 |
| `backend/` | FastAPI: run, list and compare simulations, weekly drill-down | C9 |
| `frontend/` | React interface for configuring and comparing runs | C9 |

Deployment (C9) uses Docker Compose with a backend and a frontend service. The SQLite file
is mounted into the backend container as a volume, so no database server is needed.

## Milestones

### C0: Foundation — done
- `config.yaml` with all parameters from the proposal; typed loader with overrides.
- Weekly calendar: decision date = last trading day of the week; execution at the next
  close; documents assigned to weeks by the 16:00 ET cut-off (DST-aware).
- SQLite storage: schema, upsert and query helpers.
- Unit tests, including a randomized no-look-ahead test for document assignment.

### C1: Market data
- Daily adjusted prices from Yahoo Finance for the 11 core assets, XLRE, XLC and SPY,
  2009–2023.
- DTB3 and selected macro series from FRED; daily risk-free rate.
- Daily and weekly excess returns.
- Data validation: missing days, stale prices, outlier returns, coverage report.

### C2: Optimization engine
- Ledoit–Wolf covariance on 252 days, scaled to the 4-week horizon.
- Equilibrium prior π = δΣw_ref (w_ref: 70% sectors equally weighted, 20% TLT, 10% GLD).
- Black–Litterman posterior in Woodbury form.
- Long-only mean-variance QP with turnover penalty (CVXPY, Clarabel, OSQP fallback).
- Closed-form Lagrange solution and KKT checks as solver tests.
- Baselines: equal weight, minimum variance, inverse volatility, Markowitz, BL without views.
- Tested on synthetic data, independent of C1.

### C3: Technical features and view generator
- MOM (12 weeks, skipping the latest), REV (latest week), VOL (20 days), cross-sectionally
  standardized.
- Ridge view generator for any feature source: purged expanding window (minimum 156
  weeks), view only if |q| > 0.5σ, Ω from out-of-fold residuals; builds P, q, Ω.

### C4: Replay engine and metrics
- Point-in-time data access per decision date.
- Weekly walk-forward loop: drifted weights, execution at the next close, transaction costs.
- Metrics: annualized return and volatility, Sharpe, Sortino, maximum drawdown, CVaR 95%,
  turnover. Runs saved with their configuration.
- **Result: a technical-only simulation running end to end (progress report).**

### C5: Text data
- FOMC statements, minutes and press conferences scraped from the Federal Reserve website.
- FNSPID filtered to 2010–2023, title plus first paragraph, sampled by sector.
- Cleaning, MinHash deduplication, timestamps converted to ET, tickers mapped to GICS
  sectors.

### C6: NLP
- Topic labels: LLM-assisted pre-labelling, reviewed by hand; written annotation guideline.
- Baselines: Loughran–McDonald lexicon, TF-IDF with logistic regression, pre-trained FinBERT.
- Multi-task FinBERT (topic, sentiment, stance) with masked loss; macro-F1 per task.
- Scoring of the full corpus into `doc_scores`, versioned by model.

### C7: News indices and news views
- Weekly indices with a 14-day half-life over the last 8 weeks: topic sentiment S^(k),
  monetary stance H, sector sentiment S^sec, uncertainty U.
- Rolling 156-week standardization using past values only.
- News features fed into the C3 view generator; Ω_news scaled by uncertainty.

### C8: Evaluation
- Ablation: no views, technical only, news only, both; against all baselines and SPY.
- Placebo: news indices circularly shifted (≥ 52 weeks), 200 runs.
- Stationary bootstrap for Sharpe differences (block length 8 weeks, 5,000 resamples).
- Sensitivity: τ, γ, w_max, half-life, weekly vs monthly rebalancing.
- Stress periods 2020 and 2022; NLP quality vs portfolio outcome.

### C9: Web simulator
- FastAPI: `POST /backtest`, `GET /runs`, `GET /compare`, `GET /week/{date}`.
- React interface: configuration, equity and drawdown charts, weights over time,
  run comparison, weekly drill-down.
- Docker Compose.

### C10: Reports
- Progress report, final presentation, final report.

## Schedule

| Weeks | Dates | Milestones |
|---|---|---|
| 7–8 | 30 Oct – 13 Nov | C0–C4 (minimal versions) → progress report |
| 9–10 | 16 Nov – 27 Nov | C5, C6 baselines |
| 11 | 30 Nov – 04 Dec | C6 FinBERT fine-tuning and scoring, C7 |
| 12–13 | 07 Dec – 18 Dec | C8 |
| 14–15 | 21 Dec – 01 Jan | C9, report drafting |
| 16–17 | 04 Jan – 15 Jan | Presentation, final report |

## Deviations from the proposal

| Proposal | Implementation | Reason |
|---|---|---|
| MySQL database | SQLite (single file, Python standard library) | The simulator is read-mostly and single-user; no server to run or deploy. |
| ~1,500 topic labels, 200 double-annotated | ~600–800 labels, LLM-assisted and reviewed by hand; 150–200 annotated independently by a second person for Cohen's κ | Workload; the reduced set is stratified by year, source and topic. |

## Open decisions

- [ ] Second annotator for Cohen's κ (a classmate is preferred over an LLM, which would not
      be inter-annotator agreement in the usual sense).
- [ ] FNSPID sample size per sector and year, given scoring cost.
- [ ] GPU for fine-tuning: Colab, Kaggle or cloud.
- [ ] Frontend scope: weekly drill-down may be delivered last.

## Known limitations

- **Text model and the test period.** Document scores are computed once, offline. If the
  NLP model is trained on labelled data from 2018–2023 (the FOMC stance dataset extends to
  2022), the test-period scores contain mild look-ahead. Training only on pre-2018 examples,
  or reporting both variants, will be decided in C6.
- **Domain shift.** The sentiment and stance datasets are sentence-level text from financial
  reports and FOMC documents; news headlines differ in style.
- **Independent view errors.** The block-diagonal Ω assumes news and technical view errors
  are uncorrelated, as stated in the proposal.
