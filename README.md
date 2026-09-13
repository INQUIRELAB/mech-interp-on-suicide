# Suicidality detection in large language models depends on a four-dimensional semantic subspace

[![License: INQUIRE Academic](https://img.shields.io/badge/License-INQUIRE%20Academic-red.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](INSTALL.md)

**Authors:**  
Nafiz Ahmed, Sarah S. Sharif, Dingjing Shi, and Yaser M. Banad*  
School of Electrical and Computer Engineering, University of Oklahoma, Oklahoma, United States  
School of Psychology, Georgia Institute of Technology, Atlanta, Georgia, United States  
*Corresponding author: bana@ou.edu

Code and per-claim result files for the paper of the same name
(Ahmed, Sharif, Shi & Banad).

Every number, table, and figure in the paper and its Supplementary Information is
produced by one of the scripts below and written to a named result file. The mapping is
given in [Reproducing each result](#reproducing-each-result).

## Table of contents

- [📦 What is and is not in this repository](#what-is-and-is-not-in-this-repository)
- [🚀 Quick start](#quick-start)
- [💾 Environment](#environment)
- [⚙️ Precision: read this before reproducing a number](#precision-read-this-before-reproducing-a-number)
- [📊 Data](#data)
- [🧪 Reproducing each result](#reproducing-each-result)
- [🎲 Determinism and seeds](#determinism-and-seeds)
- [🖥️ Hardware and runtime](#hardware-and-runtime)
- [🏗️ Repository layout](#repository-layout)
- [🔐 Responsible use](#responsible-use)
- [📝 License](#license)
- [📚 Citation](#citation)

<a id="what-is-and-is-not-in-this-repository"></a>
## 📦 What is and is not in this repository

Included: all analysis code, the experiment drivers, the figure-rendering script, the
environment specification, and the seeding machinery needed to reproduce the reported
numbers.

Not included, deliberately:

- **Raw corpus text.** The five datasets are redistributed by their original sources
  under their own terms; see [Data](#data). No post text is reproduced here.
- **Fine-grained model artifacts** (probe weights, SAE dictionaries, steering vectors).
  These are available to vetted researchers on request from the corresponding author.
  They are gated rather than released openly because the same direction characterized in
  the paper can be used to steer a model toward distress-affirming output. See
  [Responsible use](#responsible-use).

<a id="quick-start"></a>
## 🚀 Quick start

```bash
git clone https://github.com/INQUIRELAB/mech-interp-on-suicide.git
cd mech-interp-on-suicide
conda env create -f environment.yml                 # or: see INSTALL.md for venv/pip
conda activate mech-interp-suicide
cp .env.example .env                                # then edit: SEED, HF token, data paths
huggingface-cli login                               # Llama models are gated on the Hub
python verify_setup.py                              # checks imports, GPU, model access
```

[INSTALL.md](INSTALL.md) documents the full setup: other CUDA versions, CPU-only
installs, dataset acquisition, VRAM requirements, and troubleshooting.

Then reproduce a single claim, for example the behavioral gate:

```bash
MI_PRECISION=fp16 python analysis/sanity/sc2_binary_gate.py
# -> results/sc2_binary/gate.json
```

See [Precision](#precision-read-this-before-reproducing-a-number) for why `MI_PRECISION`
matters: without it the scripts run in 4-bit and will not match the paper.

Or the whole binary-task suite, one model in VRAM at a time:

```bash
python analysis/sanity/rerun_all_sc2.py
```

`rerun_all_sc2.py` runs each script as its own subprocess, logs stdout and stderr per
script, records timing and exit status, continues past failures, and writes a manifest
and markdown summary at the end. Use `--only NAME [NAME ...]` or `--skip NAME [NAME ...]`
to run part of the suite.

<a id="environment"></a>
## 💾 Environment

Reported results were produced with Python 3.11.13 and CUDA 12.8. `requirements.txt`
pins the exact versions used:

| Package | Version |
| --- | --- |
| torch | 2.11.0+cu128 |
| transformers | 5.16.1 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| accelerate | 1.14.0 |
| datasets | 5.0.1 |
| huggingface_hub | 1.30.0 |
| tokenizers | 0.23.2 |
| safetensors | 0.8.0 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| einops | 0.8.2 |
| tqdm | 4.70.0 |

`bitsandbytes` is required for the 4-bit and 8-bit paths, which is what the `sc2_*`
scripts use by default; see [Precision](#precision-read-this-before-reproducing-a-number).
The fp16 path does not need it.

<a id="precision-read-this-before-reproducing-a-number"></a>
## ⚙️ Precision: read this before reproducing a number

The `sc2_*` scripts were written for a 12 GB card and therefore call the loader
with `load_in_4bit=True` hardcoded. **Running a script directly reproduces the
4-bit numbers, not the fp16 numbers reported in the paper.**

`MI_PRECISION` overrides whatever the calling script asked for, and is honoured by
`src.models.ModelLoader`. To reproduce the paper:

```bash
MI_PRECISION=fp16 python analysis/sanity/sc2_binary_gate.py
```

Or run the whole paper-backing suite in fp16 with the dedicated runner, which sets
the variable for you and archives output to `results/sc2_binary_fp16/` so the 4-bit
JSONs stay intact for the precision-comparison table:

```bash
python analysis/sanity/run_fp16_verify.py              # core suite in fp16
python analysis/sanity/run_fp16_verify.py --only gate faithfulness
python analysis/sanity/run_fp16_verify.py --precision bf16
python analysis/sanity/run_fp16_verify.py --dry-run
```

Accepted values are `fp16`, `bf16`, `8bit`, `4bit`. The quantized paths additionally
require `bitsandbytes`, which is listed in `requirements.txt`; the fp16 path does not
need it, and `run_fp16_verify.py` skips the scripts whose dependencies are absent.

In the result tables below, paths are given as `results/sc2_binary/...`, which is where
a direct (4-bit) run writes. Under `run_fp16_verify.py` the same files appear in
`results/sc2_binary_fp16/`.

<a id="data"></a>
## 📊 Data

Five corpora are used as released, with no new data collection. Acquire them from their
original sources and set the paths in `.env`; `utilities/download_datasets.py` and
`utilities/cache_datasets.py` automate the public ones.

| Dataset | Role in the paper | Access |
| --- | --- | --- |
| Suicide-Watch-232K | Primary feature analysis | Public (Kaggle, CC BY-SA 4.0) |
| SDCNL | Generalization | Public (released for research) |
| SWMH | Generalization; depression class for the specificity test | Public (CC BY-NC 4.0) |
| DeepSuiMind | Implicit-intent task | Public (synthetic dialogue) |
| UMD Reddit Suicidality v2 | Expert-severity validation only | **Access-controlled**, under its data-usage agreement |

The UMD dataset cannot be redistributed. Obtain it directly from the American Association
of Suicidology under its data-usage agreement. Scripts that depend on it
(`sc2_umd_severity.py`) will exit with a clear message if the path is unset.

<a id="reproducing-each-result"></a>
## 🧪 Reproducing each result

Each row of the paper's evidence table (SI Section S8) maps to one script and one result
file. Paths are relative to the repository root. Run from the repository root.

### Main claims

| Paper claim | Command | Result file |
| --- | --- | --- |
| Behavioral gate (232K) | `python analysis/sanity/sc2_binary_gate.py` | `results/sc2_binary/gate.json` |
| Decodability (probe AUC by layer) | `python analysis/sanity/sc2b_probe_gate.py` | `results/sc2_binary/probe_gate.json` |
| Semantic, not lexical (keyword masking) | `python analysis/sanity/sc3_surface_confounds.py` | `results/sc3/confounds.json` |
| Specificity (suicide vs. depression) | `python analysis/sanity/sc2_specificity.py` | `results/sc2_binary/specificity.json` |
| Specificity, matched-negative ladder | `python analysis/sanity/gcl/run_gcl.py --concept suicide_depression` | `results/gcl/suicide_depression/llama_3_1_8b_instruct.json` |
| Necessity (per-layer ablation) | `python analysis/sanity/sc2_faithfulness_perlayer.py` | `results/sc2_binary/faithfulness_perlayer.json` |
| Necessity (single layer) | `python analysis/sanity/sc2_faithfulness.py` | `results/sc2_binary/faithfulness.json` |
| Intrinsic dimension (rank-k sweep) | `python analysis/sanity/sc2_rankk.py` | `results/sc2_binary/rankk.json` |
| Sufficiency (steering, not established) | `python analysis/sanity/sc2_sufficiency.py` | `results/sc2_binary/sufficiency.json` |
| Steering salience confound (5 concepts) | `python analysis/sanity/gcl/steering_salience.py` | `results/gcl/steering_salience.json` |
| Generalization (decode: SDCNL, SWMH) | `python analysis/sanity/sc2_generalization.py` | `results/sc2_binary/generalization.json` |
| Generalization (causal: cross-ablation) | `python analysis/sanity/sc2_crossablation.py` | `results/sc2_binary/crossablation.json` |
| Downstream detection metrics | `python analysis/sanity/sc2_downstream_table.py` | `results/sc2_binary/downstream_table.json` |
| Cross-model (Qwen) | `python analysis/sanity/sc2_secondmodel.py qwen` | `results/sc2_binary/secondmodel_qwen.json` |
| Cross-model (Mistral) | `python analysis/sanity/sc2_secondmodel.py mistral` | `results/sc2_binary/secondmodel_mistral.json` |
| Cross-scale (encoding vs. use) | `python analysis/sanity/sc2_smallmodel_control.py` | `results/sc2_binary/smallmodel_control.json` |
| Second concept (toxicity ladder) | `python analysis/sanity/gcl/run_gcl.py --concept toxicity` | `results/gcl/toxicity/*.json` |
| Framework comparison across concepts | `python analysis/sanity/gcl/gcl_compare.py` | `results/gcl/framework_comparison.json` |
| Severity axis (ordinal projection) | `python analysis/phase11_severity_axis.py` | `results/phase11/*.json` |
| Expert severity (UMD) | `python analysis/sanity/sc2_umd_severity.py` | `results/sc2_binary/umd_severity.json` |
| Implicit intent (NO-GO) | `python analysis/sanity/sc2_intent_causal.py` | `results/sc2_binary/intent_causal.json` |
| Implicit intent, positive control | `python analysis/sanity/sc2_intent_upperbound.py` | `results/sc2_binary/intent_upperbound.json` |

### Localization (Supplementary)

| Analysis | Command | Result file |
| --- | --- | --- |
| Head knockout | `python analysis/sanity/sc2_head_knockout.py --n 60` | `results/sc2_binary/head_knockout.json` |
| Head knockout, statistics and FDR | `python analysis/sanity/sc2_head_knockout_stats.py` | `results/sc2_binary/head_knockout_stats.json` |
| Head knockout, power / effect bound | `python analysis/sanity/sc2_head_knockout_power.py` | `results/sc2_binary/head_knockout_power.json` |
| Activation patching (indirect effect) | `python analysis/sanity/sc2_patching.py` | `results/sc2_binary/patching.json` |
| Attribution patching | `python analysis/sanity/sc2_attribution_patching.py` | `results/sc2_binary/attribution_patching.json` |
| Edge attribution patching | `python analysis/sanity/sc2_edge_attribution_patching.py` | `results/sc2_binary/edge_attribution_patching.json` |
| Direct logit attribution | `python analysis/sanity/sc2_direct_logit_attribution.py` | `results/sc2_binary/direct_logit_attribution.json` |
| Sparse autoencoder | `python analysis/sanity/sc2_sae.py` | `results/sc2_binary/sae.json` |

### Statistics and robustness

| Analysis | Command | Result file |
| --- | --- | --- |
| Bootstrap confidence intervals | `python analysis/sanity/sc2_ci.py` | `results/sc2_binary/bootstrap_ci.json` |
| Significance tests and BH correction | `python analysis/sanity/sc2_significance.py` | `results/sc2_binary/significance.json` |
| Precision robustness (4-bit) | `python analysis/sanity/sc2_precision_robustness.py --precision 4bit` | `results/sc2_binary/precision_4bit.json` |
| Precision robustness (8-bit) | `python analysis/sanity/sc2_precision_robustness.py --precision 8bit` | `results/sc2_binary/precision_8bit.json` |
| Precision comparison | `python analysis/sanity/sc2_precision_robustness.py --compare` | `results/sc2_binary/precision_compare.json` |
| Seed sweep (variance across seeds) | `SEEDS=0,1,2,3,4 python analysis/sanity/seed_sweep.py` | `results/sc2_binary/seed_sweep.json` |
| Worked input/output examples | `python analysis/sanity/dump_examples.py` | `results/sc2_binary/examples.json` |

### Figures

```bash
python paper/make_figures.py
```

Renders every main-text and supplementary figure from the result files above. It reads
only committed result JSON, never the model, so figures can be regenerated without a GPU.

<a id="determinism-and-seeds"></a>
## 🎲 Determinism and seeds

Seeding is centralized in `src/repro.py`. Every stochastic script calls `set_seed()` near
the top and derives local generators from the master seed rather than hardcoded
constants:

```python
from src.repro import set_seed, get_seed
set_seed()
rng = np.random.default_rng(get_seed())
boot_rng = np.random.default_rng(get_seed() + 12345)   # independent stream
```

The single source of truth is the `SEED` environment variable, read from `.env` and
defaulting to `0`, which is the value that reproduces the numbers in the paper. Set
`SEED` to a different value to re-run under another draw, or set `SEEDS=0,1,2,3,4` and
use `analysis/sanity/seed_sweep.py` for mean and standard deviation across seeds.

`DETERMINISTIC=1` additionally forces torch's deterministic algorithms. It is off by
default because it raises on some CUDA operations and slows execution; for the
inference-only work that makes up most of this project, cuDNN determinism suffices.

Directions and thresholds are fit on training splits and evaluated held out throughout.

<a id="hardware-and-runtime"></a>
## 🖥️ Hardware and runtime

All reported experiments run in fp16 on a single 96 GB GPU. An 8B model at fp16 needs
roughly 16 GB of VRAM, so the full suite does not fit on a 12 GB card; on smaller cards
use the quantized path (`--precision 4bit`) or a smaller `--model`. The quantized
replications run the same scripts under a single environment variable, with weight
folding and centering disabled, since those corrupt quantized weights.

<a id="repository-layout"></a>
## 🏗️ Repository layout

```
analysis/
  sanity/            per-claim experiment scripts (sc1-sc4, see table above)
    gcl/             generalized concept ladder: multi-concept driver and comparisons
    rerun_all_sc2.py sequential driver for the full binary-task suite
    run_all_sanity.py driver for the SC-1..SC-4 sanity checks
  phase11_severity_axis.py  severity-axis analysis
src/
  repro.py           centralized seeding; single source of truth for SEED
  models.py          model loading and precision handling
  analysis_utils.py  shared analysis helpers
  hf_adapter.py      HuggingFace interop
  hf_adapter_tl.py   TransformerLens interop
  suicide_dataset_loader.py
utilities/           dataset download/caching, annotation helpers, table generation
paper/
  make_figures.py    renders all figures from result files
results/             result files written by the scripts (created on first run)
environment.yml      conda environment specification
requirements.txt     pinned package versions (single source of truth)
INSTALL.md           full setup, CUDA variants, troubleshooting
verify_setup.py      environment, GPU, and model-access check
```

<a id="responsible-use"></a>
## 🔐 Responsible use

This code supports an audit of model behavior. It is not a clinical instrument, it was
not validated for clinical use, and it must not be deployed as a screening tool. The
paper reports classification behavior and internal representations only, follows
safe-messaging guidance throughout, and neither generates nor describes means of
self-harm. Use the corpora only under their stated terms, do not redistribute raw text,
and do not attempt re-identification.

If you or someone you know is in distress, the 988 Suicide and Crisis Lifeline (US) and
<https://findahelpline.com> list available services.

<a id="license"></a>
## 📝 License

INQUIRE Lab Academic Research License (Version 1.0) — see [LICENSE](LICENSE).

Use is limited to non-commercial academic research, peer review, and verification of the
results reported in the associated publication. Redistribution, derivative works, and any
commercial use require prior written permission from the INQUIRE Lab. Any publication
describing results obtained with this software must cite the paper below.

<a id="citation"></a>
## 📚 Citation

```bibtex
@article{ahmed2026suicidality,
  title={Suicidality detection in large language models depends on a four-dimensional semantic subspace},
  author={Ahmed, Nafiz and Sharif, Sarah S. and Shi, Dingjing and Banad, Yaser M.},
  year={2026}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).
