# reports

Generated outputs of the technical-only simulation. Do not edit by hand; regenerate with:

```bash
python -m simulation.experiment      # writes results/ (2013-01-01 to 2023-12-31 by default)
python -m simulation.report          # writes figures/ and tables/
```

Use `--out DIR` on both commands to write somewhere else, for example
`reports/validation` for a run that stays inside the development period.

| Folder | Content |
|---|---|
| [results/](results/) | Raw experiment output: daily NAV, weekly returns, turnover, target weights per strategy, metrics. Column descriptions are in [simulation/README.md](../simulation/README.md#outputs-in-results). |
| [figures/](figures/) | F1 cumulative value of one dollar, F2 drawdown from prior peak, F3 rolling 52-week Sharpe ratio and volatility. Each as PDF and PNG. |
| [tables/](tables/) | T1 performance metrics and T2 calendar-year returns, each as CSV and LaTeX. |

## Read before citing

The committed files cover **2013-2023**, so they include the 2018-2023 holdout period. They
show that the pipeline works end to end; they are not an untouched final test if any result
here informed a design choice. The final evaluation is milestone C8 in [plan.md](../plan.md).
