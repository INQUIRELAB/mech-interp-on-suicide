"""TransformerLens-backed adapter exposing the same interface as HookedHFModel.

Drop-in replacement for `src.hf_adapter.HookedHFModel`: the analysis code
(phase1..phase12) imports HookedHFModel + HFModelConfig from this module and
uses `run_with_cache`, `add_hook`, `remove_hooks`, `.cfg.n_layers`, etc. — the
same surface the manual HF hook wrapper exposes.

Under the hood this is just `transformer_lens.HookedTransformer`, which already
uses the exact hook names the analysis code relies on:
    blocks.{L}.hook_attn_out
    blocks.{L}.hook_mlp_out

Usage (mirrors the HF adapter):
    from src.hf_adapter_tl import HookedHFModel, load_tl_model

    model = load_tl_model("meta-llama/Llama-3.1-8B-Instruct", quantize="4bit")
    tokens = model.tokenizer.encode("Hello", return_tensors="pt").to(model.device)
    logits, cache = model.run_with_cache(tokens)
    mlp = cache["blocks.0.hook_mlp_out"]
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformer_lens import HookedTransformer


class _CacheWithGet:
    """Thin wrapper around TL's ActivationCache adding `.get(key, default=None)`.

    Analysis code (analysis_utils, phase6, phase7) calls `cache.get(hook_name)`
    which exists on plain dicts but not TL's cache. We forward `__contains__`,
    `__getitem__`, and `__iter__` to the underlying cache.
    """

    def __init__(self, cache):
        self._cache = cache

    def get(self, key, default=None):
        if key in self._cache:
            return self._cache[key]
        return default

    def __contains__(self, key):
        return key in self._cache

    def __getitem__(self, key):
        return self._cache[key]

    def __iter__(self):
        return iter(self._cache)

    def keys(self):
        return self._cache.keys()


@dataclass
class HFModelConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_vocab: int
    n_ctx: int
    eps: float = 1e-5


class HookedHFModel:
    """TransformerLens-backed model with the HookedHFModel surface.

    Wraps a `HookedTransformer` and exposes the subset of methods the phase
    analysis code uses: `run_with_cache`, `add_hook`, `remove_hooks`, `.cfg`,
    `.tokenizer`, `.device`, and call-as-forward.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        device: Optional[torch.device] = None,
    ):
        """Accept either a HookedTransformer or a raw HF causal LM.

        The analysis code constructs this as `HookedHFModel(hf_model, tokenizer)`
        after loading via `ModelLoader`. To preserve that call site, if `model`
        is an HF model we wrap it with `HookedTransformer.from_pretrained(
        ..., hf_model=model)` — which re-hooks the existing weights in place
        rather than re-downloading.
        """
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if isinstance(model, HookedTransformer):
            tl_model = model
            self._hf = None
        else:
            # Single-copy load path. TL's `hf_model=` kwarg retains an internal
            # reference to the HF model, so the two copies coexist (~32 GB for
            # Llama-8B bf16) — OOMs on A100-40GB. We instead:
            #   1. Record the model name
            #   2. Free the HF model's GPU storage in-place
            #   3. Let TL load its own single copy via from_pretrained_no_processing
            # This matches the `else` branch in the reference impl
            # (liuqi6777/llm-relevance main.py) which loads TL without hf_model=.
            import gc as _gc
            hf_name = getattr(model.config, "_name_or_path", None) or model.config.name_or_path
            # Empty each HF parameter's storage in place — breaks any lingering
            # references (DataParallel, hooks, etc.) without destroying the
            # object graph.
            with torch.no_grad():
                for p in model.parameters():
                    p.data = torch.empty(0, dtype=p.dtype, device=p.device)
                for b in model.buffers():
                    b.data = torch.empty(0, dtype=b.dtype, device=b.device)
            del model
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            tl_dtype = "bfloat16" if str(self.device).startswith("cuda") else "float32"
            tl_model = HookedTransformer.from_pretrained_no_processing(
                hf_name,
                tokenizer=tokenizer,
                device=str(self.device),
                dtype=tl_dtype,
                n_ctx=2048,
            )
            tl_model.eval()
            self._hf = None
        self.model = tl_model  # the HookedTransformer
        self.tokenizer = tokenizer if tokenizer is not None else tl_model.tokenizer
        tlcfg = tl_model.cfg
        self.cfg = HFModelConfig(
            n_layers=tlcfg.n_layers,
            n_heads=tlcfg.n_heads,
            d_model=tlcfg.d_model,
            d_head=tlcfg.d_head,
            d_vocab=tlcfg.d_vocab,
            n_ctx=tlcfg.n_ctx,
            eps=getattr(tlcfg, "eps", 1e-5),
        )
        self._custom_hooks = []  # list of hook names added via add_hook

    @torch.no_grad()
    def run_with_cache(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with full activation cache.

        TL's `ActivationCache` supports `__contains__` and `__getitem__` but
        NOT `.get()` — analysis code (analysis_utils, phase6, phase7) uses
        `.get(hook_name)`, so we return a thin dict wrapper that adds it.
        """
        logits, cache = self.model.run_with_cache(tokens)
        return logits, _CacheWithGet(cache)

    def add_hook(self, hook_name: str, hook_fn: Callable) -> None:
        """Register a forward hook on a TL hook point.

        hook_fn signature is `(activation, hook) -> modified_activation`, exactly
        what TL expects. The existing analysis code already uses this signature.
        """
        self.model.add_hook(hook_name, hook_fn)
        self._custom_hooks.append(hook_name)

    def remove_hooks(self) -> None:
        """Clear all hooks added via add_hook (mirrors HookedHFModel)."""
        self.model.reset_hooks()
        self._custom_hooks.clear()

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(tokens)

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self

    def to(self, device):
        self.model.to(device)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        return self


# ---------------------------------------------------------------------------
# Loader helper — parallels src.models.ModelLoader but builds a TL model.
# ---------------------------------------------------------------------------

_MODEL_ALIASES = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
}


def _resolve(name: str) -> str:
    key = name.lower().replace("_", "-")
    return _MODEL_ALIASES.get(key, name)


def load_tl_model(
    model_name: str,
    quantize: Optional[str] = "auto",  # "4bit" | "8bit" | "fp16" | "auto" | None
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> HookedHFModel:
    """Load a HookedTransformer wrapped in the HookedHFModel interface.

    Strategy: load the HF model ourselves (so we can apply 4/8-bit quantization
    via bitsandbytes), then hand it to `HookedTransformer.from_pretrained` via
    the `hf_model=` argument. We disable weight-folding / centering because
    those transforms don't compose cleanly with quantized weights.

    Args:
        model_name: short alias or full HF repo id.
        quantize: "4bit", "8bit", "fp16", "auto", or None. "auto" picks 4-bit
            when VRAM < 16GB, 8-bit when < 24GB, else fp16. None = fp16.
        device: "cuda" or "cpu".
        cache_dir: HF cache override.
        hf_token: passed to `from_pretrained` for gated models.
    """
    hf_name = _resolve(model_name)

    # Auto-select quantization tier based on VRAM.
    if quantize == "auto":
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 16:
                quantize = "4bit"
            elif vram_gb < 24:
                quantize = "8bit"
            else:
                quantize = "fp16"
        else:
            quantize = "fp16"

    quant_cfg = None
    if quantize == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        cache_dir=cache_dir,
        padding_side="left",
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_cfg is not None:
        # GATE-01 fix: previously quant_cfg was built then discarded — the model
        # loaded in bf16 (~16GB) and OOM'd on <16GB GPUs. To honor 4/8-bit we
        # must load the HF model OURSELVES with the quantization_config, then
        # hand it to TL via `hf_model=` (TL cannot quantize on its own). This
        # path keeps a single quantized copy on the GPU.
        # NOTE: untested locally — transformer_lens is not installed in this
        # environment; the working analysis pipeline uses src.hf_adapter +
        # ModelLoader (which already applies quant_cfg correctly). Verify on a
        # machine with TL before relying on this path.
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=quant_cfg,
            device_map={"": device},
            cache_dir=cache_dir,
            token=hf_token,
            torch_dtype=torch.float16,
        )
        tl_model = HookedTransformer.from_pretrained_no_processing(
            hf_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            dtype="float16",
            n_ctx=2048,
        )
    else:
        # fp16/bf16 single-copy path: let TL load weights directly. `hf_model=`
        # would retain a reference and double VRAM, so we skip it here.
        tl_dtype = "bfloat16" if device == "cuda" else "float32"
        tl_model = HookedTransformer.from_pretrained_no_processing(
            hf_name,
            tokenizer=tokenizer,
            device=device,
            dtype=tl_dtype,
            n_ctx=2048,
        )
    tl_model.eval()

    return HookedHFModel(tl_model, tokenizer=tokenizer, device=torch.device(device))
