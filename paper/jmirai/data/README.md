# Result files for the JMIR AI paper

Every number in the paper, its figures and its appendix tables is read from these files by the
`make_*.py` scripts one level up. They hold aggregate statistics and per-post model scores; they hold
no post text.

Not included:

- per-user score files for the UMD Reddit Suicidality Dataset (`umd_v2/*.npz`, `yes_bias/*.npz`),
  because they carry dataset identifiers and expert labels;
- `paper2/gate/eval_rebuilt.json`, because it contains post text.

The summaries in `umd_v2/` and `yes_bias/` are derived from the UMD dataset and may not be
redistributed. Access to the dataset itself is by application to its maintainers at the
University of Maryland.
