"""Figures and LaTeX tables for the report, made from the files of ``simulation.experiment``.

    python -m simulation.report                  # reads reports/results, writes reports/figures, reports/tables

F1  cumulative value of 1 invested (log scale)
F2  drawdown from the running peak
F3  rolling 52-week Sharpe ratio and volatility
T1  performance metrics over the whole period
T2  calendar-year returns
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.config import PROJECT_ROOT, load_config  # noqa: E402

MAIN = "BL + technical views"
BENCHMARK = "SPY buy-and-hold"

# fixed categorical order (validated palette, light surface); the benchmark is neutral
COLORS = {
    "BL + technical views": "#2a78d6",
    "BL (no views)": "#eb6834",
    "Markowitz": "#1baf7a",
    "Min variance": "#eda100",
    "Inverse volatility": "#e87ba4",
    "Equal weight": "#6250d6",
    "SPY buy-and-hold": "#5f5e5a",
}
SHORT = {
    "BL + technical views": "BL+Tech",
    "BL (no views)": "BL",
    "Markowitz": "Markowitz",
    "Min variance": "MinVar",
    "Inverse volatility": "InvVol",
    "Equal weight": "EW",
    "SPY buy-and-hold": "SPY",
}
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
WIDTH = 6.3  # inches, about \textwidth of an A4 article

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": TEXT,
    "ytick.labelcolor": TEXT,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------- data

def load_results(folder: Path) -> dict[str, pd.DataFrame]:
    read = lambda name: pd.read_csv(folder / name, index_col=0, parse_dates=True)  # noqa: E731
    return {
        "nav": read("nav_daily.csv"),
        "weekly": read("returns_weekly.csv"),
        "rf": read("rf_weekly.csv")["rf"],
        "metrics": pd.read_csv(folder / "metrics.csv", index_col=0),
    }


def order(columns) -> list[str]:
    known = [name for name in COLORS if name in columns]
    return known + [c for c in columns if c not in COLORS]


# ---------------------------------------------------------------- figures

def _style(name: str) -> dict:
    if name == MAIN:
        return {"color": COLORS[name], "linewidth": 1.9, "zorder": 5}
    if name == BENCHMARK:
        return {"color": COLORS[name], "linewidth": 1.1, "linestyle": (0, (4, 2)), "zorder": 4}
    return {"color": COLORS.get(name, MUTED), "linewidth": 1.0, "alpha": 0.9, "zorder": 3}


def _split_marker(ax, test_start: pd.Timestamp, index: pd.DatetimeIndex, label: bool = True) -> None:
    if not index[0] < test_start < index[-1]:
        return
    ax.axvline(test_start, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
    if label:
        trans = ax.get_xaxis_transform()
        ax.text(test_start, 1.0, "validation  ", transform=trans, ha="right", va="bottom",
                color=MUTED, fontsize=7.5)
        ax.text(test_start, 1.0, "  test", transform=trans, ha="left", va="bottom",
                color=MUTED, fontsize=7.5)


def _legend(fig, ax, names) -> None:
    handles = [ax.get_lines()[i] for i in range(len(names))]
    fig.legend(handles, names, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.2, columnspacing=1.4)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def fig_cumulative(nav: pd.DataFrame, test_start, out: Path) -> None:
    names = order(nav.columns)
    growth = nav[names] / nav[names].iloc[0]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    for name in names:
        ax.plot(growth.index, growth[name], label=name, **_style(name))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 4.0, 5.0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Value of 1 invested (log scale)")
    ax.margins(x=0.01)
    _split_marker(ax, test_start, growth.index)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _legend(fig, ax, names)
    _save(fig, out, "f1_cumulative_value")


def fig_drawdown(nav: pd.DataFrame, test_start, out: Path) -> None:
    names = order(nav.columns)
    dd = nav[names] / nav[names].cummax() - 1
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    for name in names:
        ax.plot(dd.index, dd[name], label=name, **_style(name))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Drawdown from peak")
    ax.margins(x=0.01)
    _split_marker(ax, test_start, dd.index)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _legend(fig, ax, names)
    _save(fig, out, "f2_drawdown")


def rolling_stats(weekly: pd.DataFrame, rf: pd.Series, window: int = 52
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    excess = weekly.sub(rf.reindex(weekly.index), axis=0)
    roll = excess.rolling(window)
    sharpe = roll.mean() / roll.std() * np.sqrt(52)
    vol = weekly.rolling(window).std() * np.sqrt(52)
    return sharpe.dropna(how="all"), vol.dropna(how="all")


def fig_rolling(weekly: pd.DataFrame, rf: pd.Series, test_start, out: Path) -> None:
    names = order(weekly.columns)
    sharpe, vol = rolling_stats(weekly[names], rf)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 4.6), sharex=True)
    for name in names:
        ax1.plot(sharpe.index, sharpe[name], label=name, **_style(name))
        ax2.plot(vol.index, vol[name], label=name, **_style(name))
    ax1.axhline(0, color=MUTED, linewidth=0.8)
    ax1.set_ylabel("Sharpe ratio")
    ax1.set_title("Rolling 52-week Sharpe ratio", loc="left", color=TEXT)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.set_ylabel("Annualized volatility")
    ax2.set_title("Rolling 52-week volatility", loc="left", color=TEXT)
    for ax in (ax1, ax2):   # split line only: the panel titles sit where the labels would
        ax.margins(x=0.01)
        _split_marker(ax, test_start, sharpe.index, label=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _legend(fig, ax1, names)
    _save(fig, out, "f3_rolling_sharpe_volatility")


# ---------------------------------------------------------------- tables

METRIC_COLUMNS = [  # (metrics.csv column, header, kind, better)
    ("ann_return", "Return", "pct", "max"),
    ("ann_vol", "Volatility", "pct", "min"),
    ("sharpe", "Sharpe", "num", "max"),
    ("sortino", "Sortino", "num", "max"),
    ("max_drawdown", "Max DD", "pct", "min"),
    ("cvar_95", "CVaR 95\\%", "pct", "min"),
    ("ann_turnover", "Turnover", "num", "min"),
]


def _fmt(value: float, kind: str) -> str:
    if pd.isna(value):
        return "--"
    return f"{100 * value:.1f}" if kind == "pct" else f"{value:.2f}"


def table_metrics(metrics: pd.DataFrame, period: str, out: Path) -> None:
    names = order(metrics.index)
    m = metrics.loc[names]
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "t1_metrics.csv")

    best = {}
    for col, _, kind, better in METRIC_COLUMNS:
        candidates = m[col].drop(BENCHMARK, errors="ignore")
        best[col] = candidates.idxmax() if better == "max" else candidates.idxmin()

    header = " & ".join(["Strategy"] + [h for _, h, _, _ in METRIC_COLUMNS])
    rows = []
    for name in names:
        cells = []
        for col, _, kind, _ in METRIC_COLUMNS:
            text = _fmt(m.loc[name, col], kind)
            cells.append(f"\\textbf{{{text}}}" if best[col] == name else text)
        if name == BENCHMARK:
            rows.append("\\midrule")
        rows.append(" & ".join([name] + cells) + " \\\\")

    tex = "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{Performance of the main method and the baselines, weekly rebalancing, "
        f"{period}. Return, volatility, maximum drawdown and weekly CVaR in \\%; "
        f"turnover is the annual sum of $\\lVert w - \\tilde w \\rVert_1$. "
        f"Best value among the strategies (excluding SPY) in bold.}}",
        "\\label{tab:metrics}",
        "\\begin{tabular}{l" + "r" * len(METRIC_COLUMNS) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    (out / "t1_metrics.tex").write_text(tex + "\n")


def calendar_year_returns(nav: pd.DataFrame) -> pd.DataFrame:
    year_end = nav.groupby(nav.index.year).last()
    start = nav.iloc[0]
    previous = pd.concat([start.to_frame().T, year_end.iloc[:-1]])
    previous.index = year_end.index
    return year_end / previous - 1


def table_annual(nav: pd.DataFrame, out: Path) -> None:
    names = order(nav.columns)
    annual = calendar_year_returns(nav[names])
    annual.to_csv(out / "t2_annual_returns.csv")

    header = " & ".join(["Year"] + [SHORT.get(n, n) for n in names])
    rows = []
    for year, row in annual.iterrows():
        best = row.drop(BENCHMARK, errors="ignore").idxmax()
        cells = [f"\\textbf{{{_fmt(v, 'pct')}}}" if n == best else _fmt(v, "pct")
                 for n, v in row.items()]
        rows.append(" & ".join([str(year)] + cells) + " \\\\")

    legend = "; ".join(f"{SHORT[n]} = {n}" for n in names if n in SHORT and SHORT[n] != n)
    tex = "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{Calendar-year returns (\\%), after transaction costs. Best strategy of "
        f"each year (excluding SPY) in bold. {legend}.}}",
        "\\label{tab:annual}",
        "\\begin{tabular}{l" + "r" * len(names) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    (out / "t2_annual_returns.tex").write_text(tex + "\n")


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Figures and LaTeX tables for the report.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports"),
                        help="folder that contains results/ (figures/ and tables/ are written here)")
    args = parser.parse_args(argv)
    root = Path(args.out)

    data = load_results(root / "results")
    nav, weekly, rf = data["nav"], data["weekly"], data["rf"]
    test_start = pd.Timestamp(cfg.periods.test_start)
    period = f"{weekly.index[0]:%d/%m/%Y}--{weekly.index[-1]:%d/%m/%Y}"

    fig_cumulative(nav, test_start, root / "figures")
    fig_drawdown(nav, test_start, root / "figures")
    fig_rolling(weekly, rf, test_start, root / "figures")
    table_metrics(data["metrics"], period, root / "tables")
    table_annual(nav, root / "tables")
    print(f"figures -> {root / 'figures'}\ntables  -> {root / 'tables'}")


if __name__ == "__main__":
    main()
