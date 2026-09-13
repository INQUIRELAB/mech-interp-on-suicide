"""Model loading utilities for Llama, Qwen, and Mistral models.

Supports:
- meta-llama/Llama-3.1-8B-Instruct
- Qwen/Qwen2.5-7B
- mistralai/Mistral-7B-v0.1 (or instruct variant)

DOWNLOAD NOTES:
- Models are downloaded automatically on first use from HuggingFace Hub
- Cached in: ~/.cache/huggingface/hub/ (can set HF_HOME env var to change)
- First download requires internet; subsequent loads use cache
- Llama-3.1 requires HuggingFace authentication (gated model)

Model sizes (approximate):
- Full FP16: ~16 GB
- 8-bit quantized: ~8 GB
- 4-bit quantized: ~4 GB

VRAM REQUIREMENTS (for common GPUs):
- RTX 3060 Ti 12GB: Use 4-bit (fits easily) or 8-bit (tight but works)
- RTX 3090/4090 24GB: Use 8-bit comfortably
- For FP16: Need 16GB+ VRAM (A100, H100, or multiple GPUs)
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # deimos venv ships without python-dotenv; env vars still work
    pass

# Seed python/numpy/torch from the SEED env var (default 0) for every script
# that loads a model. Centralized in src.repro; see .env.example.
from src.repro import set_seed
set_seed()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


def check_vram():
    """Check available GPU VRAM and print recommendations."""
    if not torch.cuda.is_available():
        print("No CUDA GPU detected. Models will run on CPU (very slow).")
        return {"device": "cpu", "vram_gb": 0}

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
    free_gb = total_gb - allocated_gb

    print(f"GPU: {props.name}")
    print(f"Total VRAM: {total_gb:.1f} GB")
    print(f"Free VRAM: {free_gb:.1f} GB")

    # Recommendations
    print("\nRecommendations:")
    if total_gb < 8:
        print("  Your GPU has limited VRAM. Use load_in_4bit=True")
        recommended = "4bit"
    elif total_gb < 16:
        print("  Use load_in_4bit=True (safest) or load_in_8bit=True (if careful)")
        recommended = "4bit"
    elif total_gb < 24:
        print("  Use load_in_8bit=True for best balance")
        recommended = "8bit"
    else:
        print("  You can use FP16 (full precision) if desired")
        recommended = "fp16"

    return {
        "device": "cuda",
        "vram_gb": total_gb,
        "free_gb": free_gb,
        "name": props.name,
        "recommended": recommended
    }


class ModelLoader:
    """Load and manage transformer models from HuggingFace."""

    # Model name mappings
    MODELS = {
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    }

    # Approximate model sizes for user reference
    MODEL_SIZES = {
        "meta-llama/Llama-3.1-8B-Instruct": "~16 GB (FP16), ~4 GB (4-bit)",
        "Qwen/Qwen2.5-7B": "~14 GB (FP16), ~4 GB (4-bit)",
        "mistralai/Mistral-7B-v0.1": "~14 GB (FP16), ~4 GB (4-bit)",
    }

    def __init__(self, device: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            cache_dir: Override default HuggingFace cache location
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir

    def resolve_model_name(self, name: str) -> str:
        """Resolve shorthand model names to full HuggingFace paths."""
        name_lower = name.lower().replace("_", "-")
        if name_lower in self.MODELS:
            return self.MODELS[name_lower]
        return name  # Assume it's already a full path

    def get_model_info(self, model_name: str) -> dict:
        """Get information about a model."""
        hf_name = self.resolve_model_name(model_name)
        return {
            "name": hf_name,
            "size": self.MODEL_SIZES.get(hf_name, "Unknown"),
            "gated": "llama" in hf_name.lower(),  # Llama models require auth
        }

    def download_model(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        **kwargs
    ) -> str:
        """Pre-download a model to cache without loading it.

        Use this to download models in advance (e.g., on a server with good
        internet) before running experiments.

        Args:
            model_name: Short name (e.g., 'llama-3.1-8b') or full HF path
            trust_remote_code: Allow custom model architectures
            **kwargs: Additional args passed to snapshot_download

        Returns:
            Path to cached model directory
        """
        hf_name = self.resolve_model_name(model_name)
        info = self.get_model_info(model_name)

        print(f"Downloading model: {hf_name}")
        print(f"Estimated size: {info['size']}")
        if info['gated']:
            print("NOTE: This is a gated model. Ensure you've run 'huggingface-cli login'")
            print("      or set the HF_TOKEN environment variable.")

        cache_path = snapshot_download(
            hf_name,
            cache_dir=self.cache_dir,
            **kwargs
        )
        print(f"Model cached at: {cache_path}")
        return cache_path

    def load_model(
        self,
        model_name: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        auto_quantize: bool = True,
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = True,
        **kwargs
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a model and tokenizer.

        Downloads automatically on first use. Models are cached in
        ~/.cache/huggingface/hub/ (or HF_HOME env var).

        Args:
            model_name: Short name (e.g., 'llama-3.1-8b') or full HF path
            load_in_8bit: Use 8-bit quantization to save VRAM
            load_in_4bit: Use 4-bit quantization to save VRAM (recommended)
            auto_quantize: Auto-detect best quantization (default True)
            device_map: Device mapping strategy ('auto', 'cuda', etc.)
            trust_remote_code: Allow custom model architectures
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            Tuple of (model, tokenizer)
        """
        hf_name = self.resolve_model_name(model_name)
        info = self.get_model_info(model_name)

        # MI_PRECISION overrides whatever the calling script asked for. The sc2_*
        # scripts hardcode load_in_4bit=True (written for a 12 GB 3060); on a box
        # with enough VRAM we want the same code to run in fp16 without editing
        # eight call sites. Values: fp16 | bf16 | 8bit | 4bit.
        forced = os.environ.get("MI_PRECISION", "").strip().lower()
        if forced:
            if forced not in ("fp16", "bf16", "8bit", "4bit"):
                raise ValueError(f"MI_PRECISION must be fp16|bf16|8bit|4bit, got {forced!r}")
            load_in_4bit = (forced == "4bit")
            load_in_8bit = (forced == "8bit")
            auto_quantize = False
            if forced == "bf16":
                kwargs.setdefault("torch_dtype", torch.bfloat16)
            elif forced == "fp16":
                kwargs.setdefault("torch_dtype", torch.float16)
            print(f"MI_PRECISION={forced} (overriding caller's quantization args)")

        print(f"Loading model: {hf_name}")
        print(f"Device: {self.device}, Size: {info['size']}")

        if info['gated']:
            print("NOTE: Llama models require HuggingFace authentication.")
            print("      Run: huggingface-cli login")

        # Auto-detect quantization if enabled and not explicitly set
        if auto_quantize and not load_in_8bit and not load_in_4bit:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 16:
                    print(f"Auto-selected 4-bit quantization ({vram_gb:.0f}GB VRAM detected)")
                    load_in_4bit = True
                elif vram_gb < 24:
                    print(f"Auto-selected 8-bit quantization ({vram_gb:.0f}GB VRAM detected)")
                    load_in_8bit = True

        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        cache_dir = self.cache_dir or kwargs.pop("cache_dir", None)

        # Pop the dtype before the tokenizer call: **kwargs goes to both
        # from_pretrained calls, and AutoTokenizer rejects a dtype argument.
        default_dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_dtype = kwargs.pop("torch_dtype", None) or kwargs.pop("dtype", None) or default_dtype

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            padding_side="left",
            **kwargs
        )

        # Set pad token if not present (common for LLaMA models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model. transformers>=5 renamed torch_dtype to dtype; deimos runs
        # 5.16.1 while the Colab/local envs are on 4.x, so pick the name the
        # installed version actually accepts.
        import transformers as _tf
        dtype_kw = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            cache_dir=cache_dir,
            device_map=device_map if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
            **{dtype_kw: model_dtype},
            **kwargs
        )

        if device_map is None and self.device != "cuda":
            model = model.to(self.device)

        model.eval()
        print(f"Model loaded successfully!")

        return model, tokenizer

    def load_llama(self, auto_quantize: bool = True, **kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Convenience method to load Llama-3.1-8B-Instruct.

        For RTX 3060 Ti 12GB: Uses 4-bit quantization automatically.

        Requires HuggingFace authentication:
            huggingface-cli login
        """
        return self.load_model("llama-3.1-8b-instruct", auto_quantize=auto_quantize, **kwargs)

    def load_qwen(self, auto_quantize: bool = True, **kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Convenience method to load Qwen2.5-7B (open, no auth required).

        For RTX 3060 Ti 12GB: Uses 4-bit quantization automatically.
        """
        return self.load_model("qwen2.5-7b", auto_quantize=auto_quantize, **kwargs)

    def load_mistral(self, auto_quantize: bool = True, **kwargs) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Convenience method to load Mistral-7B (open, no auth required).

        For RTX 3060 Ti 12GB: Uses 4-bit quantization automatically.
        """
        return self.load_model("mistral-7b", auto_quantize=auto_quantize, **kwargs)


def list_available_models() -> dict[str, str]:
    """Return a dict of available model shorthand -> full names."""
    return dict(ModelLoader.MODELS)


def print_model_info():
    """Print information about all available models."""
    print("Available Models:")
    print("=" * 60)
    loader = ModelLoader()
    for short, full in ModelLoader.MODELS.items():
        info = loader.get_model_info(short)
        gated = " [GATED - requires auth]" if info['gated'] else ""
        print(f"\n{short}{gated}")
        print(f"  Full name: {full}")
        print(f"  Size: {info['size']}")
    print("\n" + "=" * 60)
    print("\nDownload/Cache Location:")
    print("  Default: ~/.cache/huggingface/hub/")
    print("  Change:  Set HF_HOME environment variable")
    print("\nAuthentication (for Llama):")
    print("  1. Get token: https://huggingface.co/settings/tokens")
    print("  2. Run: huggingface-cli login")
    print("  3. Or:  export HF_TOKEN=your_token_here")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        # Download mode: python models.py --download llama-3.1-8b
        model_name = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5-7b"
        loader = ModelLoader()
        loader.download_model(model_name)
    else:
        print_model_info()
