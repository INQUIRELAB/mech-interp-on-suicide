# Installation and setup

Full setup for reproducing the results in *Suicidality detection in large language models
depends on a four-dimensional semantic subspace*. For what each script reproduces, see the
tables in [README.md](README.md#reproducing-each-result).

Setup is four steps: environment, credentials, data, verification. Budget about 20 minutes
plus model download time.

## 1. Requirements

| | Reported configuration | Minimum workable |
| --- | --- | --- |
| Python | 3.11.13 | 3.10-3.12 |
| CUDA | 12.8 | 11.8 (change the torch index, below) |
| GPU | single 96 GB card | 12 GB, quantized path only |
| Disk | ~60 GB for model weights | ~20 GB for the 8B model alone |
| OS | Linux | Windows and macOS work; `bitsandbytes` is Linux-first |

An 8B model at fp16 needs roughly 16 GB of VRAM. **The fp16 numbers reported in the paper
cannot be reproduced on a 12 GB card** — below 16 GB you are restricted to the 4-bit path,
which reproduces every qualitative conclusion but overstates single-direction necessity
(ablated AUC 0.66 against 0.83 in fp16). See [Precision](README.md#precision-read-this-before-reproducing-a-number).

No GPU at all is enough to re-render every figure from the committed result files, and to
run the lexical baselines. It is not enough for any analysis that runs a model.

## 2. Environment

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate mech-interp-suicide
```

`environment.yml` pins Python and installs everything else from `requirements.txt`, so the
two install paths cannot drift apart.

To rebuild after changing `requirements.txt`:

```bash
conda env update -f environment.yml --prune
```

To remove it entirely:

```bash
conda deactivate && conda env remove -n mech-interp-suicide
```

### Option B — venv and pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

### A different CUDA version, or no GPU

`requirements.txt` pins `torch==2.11.0+cu128`. That exact wheel exists only for CUDA 12.8.
Install torch first, then the rest:

```bash
# CUDA 12.1
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121
# CPU only (figures and lexical baselines; no model analyses)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt          # torch line is already satisfied
```

Check which CUDA your driver supports with `nvidia-smi` (top right of the output).

`bitsandbytes` is needed only for the 4-bit and 8-bit paths. If it fails to build, drop it
and run in fp16 — nothing in the reported results depends on it.

## 3. Credentials and configuration

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Purpose |
| --- | --- |
| `SEED` | Master seed. `0` reproduces the paper. |
| `HF_TOKEN` | Required — the Llama checkpoints are gated on the Hub. |
| `MI_PRECISION` | Set to `fp16` to reproduce the paper. Without it the `sc2_*` scripts run 4-bit. |
| `HF_HOME` | Optional. Relocate the model cache if your home partition is small. |

`.env` is gitignored and must never be committed.

The Llama checkpoints additionally require accepting Meta's licence on the Hub while signed
in to the account that owns the token:

```bash
huggingface-cli login             # or rely on HF_TOKEN in .env
```

Request access at `huggingface.co/meta-llama/Llama-3.1-8B-Instruct`. Approval is usually
quick but is not instant; until it lands, every Llama script fails with a 403.

## 4. Data

Five corpora, none redistributed here. Four are public:

```bash
python utilities/download_datasets.py       # fetches the public corpora
python utilities/cache_datasets.py          # pre-tokenizes and caches them
```

The fifth, the UMD Reddit Suicidality Dataset v2, is access-controlled and must be obtained
directly from the American Association of Suicidology under its data-usage agreement. It is
used only for the expert-severity validation; `sc2_umd_severity.py` exits with a clear
message if its path is unset. Every other result runs without it.

Set the resulting paths in `.env`. Dataset roles are tabulated in
[README.md](README.md#data).

## 5. Verify

```bash
python verify_setup.py
```

This checks imports, GPU visibility and VRAM, model access, and dataset paths, and reports
what is missing rather than failing on the first problem. Resolve anything it flags before
running an analysis — the experiment scripts assume a working environment and their failures
are less legible.

Then reproduce one claim end to end:

```bash
MI_PRECISION=fp16 python analysis/sanity/sc2_binary_gate.py
# -> results/sc2_binary/gate.json
```

Expect a few minutes on a modern GPU, most of it model loading. If the JSON appears and its
AUC matches the behavioural gate reported in the paper, the environment is correct.

To run the whole paper-backing suite in fp16, one model in VRAM at a time:

```bash
python analysis/sanity/run_fp16_verify.py
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `403` / `GatedRepoError` on a Llama model | Licence not accepted, or `HF_TOKEN` belongs to a different account. Accept on the Hub, then re-login. |
| `CUDA out of memory` | fp16 8B needs ~16 GB. Use `MI_PRECISION=4bit`, or a smaller `--model`, or a larger card. |
| Numbers don't match the paper | `MI_PRECISION` unset, so the script ran 4-bit. Re-run with `MI_PRECISION=fp16`. |
| `ModuleNotFoundError: src` | Run from the repository root, not from inside `analysis/`. |
| `bitsandbytes` fails to import or build | Only the quantized paths need it. Run fp16, or install a build matching your CUDA. |
| `sc2_umd_severity.py` exits immediately | Expected without the access-controlled UMD dataset. |
| Torch installs a CPU build on a GPU machine | The `--extra-index-url` was omitted. Reinstall torch from the CUDA index shown above. |
| conda resolve takes very long | Use `conda env create -f environment.yml --solver=libmamba`, or mamba. |

## Reproducibility notes

Seeding is centralized in `src/repro.py` and driven by `SEED`; `0` is the value behind the
reported numbers. Directions and thresholds are fit on training splits and evaluated held
out throughout. `DETERMINISTIC=1` forces torch's deterministic algorithms — off by default
because it raises on some CUDA operations and costs speed, and cuDNN determinism is enough
for inference-only work. For spread across seeds, set `SEEDS=0,1,2,3,4` and run
`analysis/sanity/seed_sweep.py`.

## Responsible use

This code supports an audit of model behaviour. It is not a clinical instrument, was not
validated for clinical use, and must not be deployed as a screening tool. Use the corpora
only under their stated terms, do not redistribute raw text, and do not attempt
re-identification. See [Responsible use](README.md#responsible-use).
