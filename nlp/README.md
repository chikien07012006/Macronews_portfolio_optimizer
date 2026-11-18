# nlp

Placeholder for the news path. The package is empty until milestones C5-C7 in
[plan.md](../plan.md); the simulation currently runs on technical views only.

Planned contents:

- topic labelling and text baselines;
- multi-task FinBERT (sentiment and hawkish/dovish stance), fine-tuned on Financial PhraseBank
  and the Shah et al. (2023) FOMC dataset;
- scoring of FOMC communications and FNSPID financial news;
- exponentially decayed news indices per sector, standardized over a trailing window
  (`news.half_life_days`, `news.lookback_weeks`, `news.zscore_weeks` in
  [config.yaml](../config.yaml));
- news features fed to the same Ridge view generator as the technical features
  ([simulation/views.py](../simulation/views.py)).

Documents are assigned to decision weeks with `common.calendar.assign_week`: anything
published after 16:00 ET on a decision date belongs to the next week.

Dependencies (PyTorch, Transformers) are in [requirements-nlp.txt](../requirements-nlp.txt)
and are only needed for fine-tuning or scoring, typically on a cloud GPU.
