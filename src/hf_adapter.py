"""HuggingFace Model Adapter for Mechanistic Interpretability.

Wraps a HuggingFace transformer model to expose the interface expected by
analysis functions (run_with_cache, add_hook, remove_hooks).

This enables seamless use of HF models (Llama, Qwen, Mistral, etc.) with
the same circuit analysis tools as HookedTransformer.

Example:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.hf_adapter import HookedHFModel

    model_hf, tokenizer = load_model("meta-llama/Llama-3.1-8B-Instruct")
    model = HookedHFModel(model_hf, tokenizer, device="cuda")

    tokens = tokenizer.encode("Hello", return_tensors="pt").to("cuda")
    logits, cache = model.run_with_cache(tokens)
    # cache keys: "blocks.0.hook_attn_out", "blocks.0.hook_mlp_out", etc.
"""

from typing import Optional, Callable, Dict, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class HFModelConfig:
    """Configuration extracted from HuggingFace model."""
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_vocab: int
    n_ctx: int  # context window
    eps: float = 1e-5


class HookedHFModel:
    """Wraps a HuggingFace transformer model to expose HookedTransformer-like interface.

    Registers PyTorch forward hooks on attention and MLP modules to capture
    intermediate activations. Maps hook names to HookedTransformer conventions.

    Supported architectures:
        - LlamaForCausalLM (meta-llama/Llama-3.1-8B-Instruct)
        - Qwen2ForCausalLM (Qwen/Qwen2.5-7B)
        - MistralForCausalLM (mistralai/Mistral-7B-v0.1)

    All three architectures follow a standard pattern:
        model.model.layers[L].self_attn -> attention
        model.model.layers[L].mlp -> feed-forward
        model.model.norm -> final layer norm
    """

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """Initialize the adapter.

        Args:
            hf_model: Loaded HuggingFace transformer model
            tokenizer: Associated tokenizer
            device: Device to run on (cuda/cpu)
        """
        self.model = hf_model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Extract config from HF model
        self.cfg = self._extract_config(hf_model)

        # Hook storage
        self._hooks = {}  # {hook_name: (hook_id, module_ref)}
        self._cache = {}
        self._hook_handles = []

    def _extract_config(self, hf_model: nn.Module) -> HFModelConfig:
        """Extract architecture info from HuggingFace model."""
        hf_cfg = hf_model.config

        # Standard HF config names
        n_layers = hf_cfg.num_hidden_layers
        n_heads = hf_cfg.num_attention_heads
        d_model = hf_cfg.hidden_size
        d_head = d_model // n_heads

        # Vocabulary size
        d_vocab = hf_cfg.vocab_size

        # Context window (max position embeddings)
        if hasattr(hf_cfg, "max_position_embeddings"):
            n_ctx = hf_cfg.max_position_embeddings
        else:
            n_ctx = 2048  # default fallback

        return HFModelConfig(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_head=d_head,
            d_vocab=d_vocab,
            n_ctx=n_ctx,
        )

    @torch.no_grad()
    def run_with_cache(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run model forward pass and cache intermediate activations.

        Args:
            tokens: [batch, pos] token ids

        Returns:
            logits: [batch, pos, d_vocab] model output logits
            cache: Dict mapping hook names to activation tensors
        """
        self._cache.clear()

        # Register hooks to capture activations
        self._register_hooks()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(tokens, output_hidden_states=True)

        # Get logits
        logits = outputs.logits  # [batch, pos, d_vocab]

        # Retrieve cache
        cache = dict(self._cache)

        # Clean up hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        return logits, cache

    def _register_hooks(self):
        """Register forward hooks on all attention and MLP modules."""
        try:
            # Llama / Qwen / Mistral pattern: model.model.layers[L]
            layers = self.model.model.layers
        except AttributeError:
            raise RuntimeError("Unsupported model architecture. Expected model.model.layers")

        for layer_idx, layer_module in enumerate(layers):
            # Attention hook
            if hasattr(layer_module, "self_attn"):
                attn_module = layer_module.self_attn
                handle = attn_module.register_forward_hook(
                    self._make_attn_hook(layer_idx)
                )
                self._hook_handles.append(handle)

            # MLP hook
            if hasattr(layer_module, "mlp"):
                mlp_module = layer_module.mlp
                handle = mlp_module.register_forward_hook(
                    self._make_mlp_hook(layer_idx)
                )
                self._hook_handles.append(handle)

    def _make_attn_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for attention module."""
        def hook(module, input, output):
            # output is typically (attn_output, None) or just attn_output
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output

            # Store: [batch, pos, d_model]
            hook_name = f"blocks.{layer_idx}.hook_attn_out"
            self._cache[hook_name] = attn_out.detach()

        return hook

    def _make_mlp_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for MLP module."""
        def hook(module, input, output):
            # output is the MLP output [batch, pos, d_model]
            hook_name = f"blocks.{layer_idx}.hook_mlp_out"
            self._cache[hook_name] = output.detach()

        return hook

    def add_hook(self, hook_name: str, hook_fn: Callable) -> None:
        """Add a custom hook to patch activations.

        Args:
            hook_name: Hook point name (e.g., "blocks.5.hook_attn_out")
            hook_fn: Function(activation, hook) -> modified_activation

        Hook names follow pattern: "blocks.{L}.hook_{component}"
        - "blocks.{L}.hook_attn_out": Output of attention layer L
        - "blocks.{L}.hook_mlp_out": Output of MLP layer L
        - "blocks.{L}.hook_resid_pre": Input to layer L (residual stream)
        """
        # Parse hook name
        parts = hook_name.split(".")
        if len(parts) < 2 or parts[0] != "blocks":
            raise ValueError(f"Invalid hook name: {hook_name}")

        layer_idx = int(parts[1])
        component = parts[2] if len(parts) >= 3 else None

        try:
            layer_module = self.model.model.layers[layer_idx]
        except IndexError:
            raise ValueError(f"Layer {layer_idx} out of range (model has {self.cfg.n_layers} layers)")

        # Determine target module and hook type
        if "attn" in hook_name:
            target_module = layer_module.self_attn
        elif "mlp" in hook_name:
            target_module = layer_module.mlp
        else:
            raise ValueError(f"Unsupported hook target in: {hook_name}")

        # Register forward pre-hook or post-hook
        if "pre" in hook_name:
            handle = target_module.register_forward_pre_hook(
                self._make_patch_pre_hook(hook_fn, hook_name)
            )
        else:
            handle = target_module.register_forward_hook(
                self._make_patch_post_hook(hook_fn, hook_name)
            )

        self._hooks[hook_name] = (handle, target_module)

    def _make_patch_pre_hook(self, hook_fn: Callable, hook_name: str) -> Callable:
        """Wrap pre-hook function to modify input."""
        def pre_hook(module, input):
            modified = hook_fn(input[0], type("Hook", (), {"name": hook_name})())
            return (modified,) + input[1:]
        return pre_hook

    def _make_patch_post_hook(self, hook_fn: Callable, hook_name: str) -> Callable:
        """Wrap post-hook function to modify output."""
        def post_hook(module, input, output):
            hook_obj = type("Hook", (), {"name": hook_name})()
            result = hook_fn(output, hook_obj)
            # Handle attention modules that return tuples
            if isinstance(output, tuple) and not isinstance(result, tuple):
                return (result,) + output[1:]
            return result
        return post_hook

    def remove_hooks(self) -> None:
        """Remove all registered custom hooks."""
        for hook_name, (handle, module) in self._hooks.items():
            handle.remove()
        self._hooks.clear()

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convenience method to run forward pass without caching."""
        with torch.no_grad():
            outputs = self.model(tokens)
        return outputs.logits

    def parameters(self):
        """Return model parameters for optimization."""
        return self.model.parameters()

    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self

    def train(self):
        """Set model to train mode."""
        self.model.train()
        return self

    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        self.device = device
        return self
