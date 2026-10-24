"""Model backend for the concept-direction audit (Paper 1).

One place for: loading a model in fp16/bf16 (or 8-bit), building the yes/no and
multiple-choice prompts, batched last-token activation collection, batched readouts,
and forward-hook interventions on decoder-layer outputs.

Layer convention (same as the paper): hidden index j is the residual stream after
decoder block j-1 (index 0 = embeddings). A direction fit at hidden index j is applied
by hooking decoder block j-1.
"""
import contextlib
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAXLEN = 256          # body-token budget, as in the paper
BODY_RESERVE = 80     # tokens kept for the instruction/question/suffix


def _find_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "gpt_neox.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate decoder layers on this model")


def _wrap(fn):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            return (fn(out[0]),) + tuple(out[1:])
        return fn(out)
    return hook


class Backend:
    def __init__(self, model_id, dtype="float16", device="cuda", load_in_8bit=False):
        self.model_id = model_id
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"  # CPU smoke tests (CUDA_VISIBLE_DEVICES=)
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        tdt = getattr(torch, dtype)
        kw = {}
        multi = os.environ.get("AUDIT_MULTI_GPU") == "1" and torch.cuda.device_count() > 1
        if multi:   # split one model across the visible GPUs (e.g. two 16 GB cards)
            # cap each GPU at (free - headroom) so activations fit on the smaller / shared card
            head = float(os.environ.get("AUDIT_GPU_HEADROOM_GB", "2.5"))
            free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())]
            kw = {"device_map": "auto", "max_memory": {i: f"{max(f - head, 1.0):.1f}GiB" for i, f in enumerate(free)}}
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            kw = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "auto"}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=tdt, **kw)
        except TypeError:  # transformers < 4.56 uses torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=tdt, **kw)
        if not load_in_8bit and not multi:
            self.model.to(device)
        placed = set(map(str, getattr(self.model, "hf_device_map", {}).values()))
        if placed & {"cpu", "disk"}:   # silent offload makes a run ~50x slower; fail fast instead
            raise RuntimeError(f"model partly offloaded to {sorted(placed & {'cpu', 'disk'})}; "
                               f"free GPU memory or lower AUDIT_GPU_HEADROOM_GB")
        self.model.eval()
        self.dtype = tdt
        self.layers = _find_layers(self.model)
        self.n_layers = len(self.layers)
        self.use_chat = self.tok.chat_template is not None
        self.truncate_side = "right"              # set per concept by the driver
        self.body_budget = MAXLEN - BODY_RESERVE  # set per concept by the driver

    # ------------------------------------------------------------------ prompts
    def token_id(self, s):
        return self.tok.encode(s, add_special_tokens=False)[0]

    def _truncate(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        budget = self.body_budget
        ids = ids[-budget:] if self.truncate_side == "left" else ids[:budget]
        return self.tok.decode(ids)

    def _wrap_chat(self, content):
        if self.use_chat:
            return self.tok.apply_chat_template([{"role": "user", "content": content}],
                                                add_generation_prompt=True, tokenize=False)
        return content

    def polar_prompt(self, text, prefix, question):
        content = (f"{prefix}\n\nText: {self._truncate(text)}\n\n{question} "
                   f"Answer with Yes or No.\nAnswer:")
        return self._wrap_chat(content)

    def mc_prompt(self, text, options):
        """options: list of 4 description strings, already in display order."""
        letters = "ABCD"
        opts = "\n".join(f"{letters[i]}) The text {o}." for i, o in enumerate(options))
        content = (f"Read the following text.\n\nText: {self._truncate(text)}\n\n"
                   f"Which statement best describes the text?\n{opts}\n"
                   f"Answer with A, B, C or D.\nAnswer:")
        return self._wrap_chat(content)

    # ------------------------------------------------------------------ forward
    def _encode(self, prompts):
        add_special = not self.use_chat  # chat templates already contain BOS
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=add_special)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)   # correct positions under left padding
        return ids, mask, pos

    @torch.no_grad()
    def last_logits(self, prompts, batch=16):
        out = []
        for i in range(0, len(prompts), batch):
            ids, mask, pos = self._encode(prompts[i:i + batch])
            lg = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                            logits_to_keep=1).logits[:, -1, :]
            out.append(lg.float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def collect(self, prompts, yes_id, no_id, batch=16, verbose=True):
        """Return (LD [N], H [N, n_layers+1, d] float16) at the final token."""
        lds, hs = [], []
        for i in range(0, len(prompts), batch):
            ids, mask, pos = self._encode(prompts[i:i + batch])
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                             output_hidden_states=True, logits_to_keep=1)
            lg = out.logits[:, -1, :].float()
            lds.append((lg[:, yes_id] - lg[:, no_id]).cpu().numpy())
            hs.append(torch.stack([h[:, -1, :].to("cpu", torch.float16) for h in out.hidden_states], 1)
                      .numpy())
            if verbose and (i // batch) % 20 == 0:
                print(f"    collect {min(i + batch, len(prompts))}/{len(prompts)}", flush=True)
        return np.concatenate(lds), np.concatenate(hs)

    def logit_diff(self, prompts, yes_id, no_id, batch=16):
        lg = self.last_logits(prompts, batch)
        return (lg[:, yes_id] - lg[:, no_id]).numpy()

    def letter_logprobs(self, prompts, letter_ids, batch=16):
        """Log-probabilities renormalized over the four answer letters. [N, 4]."""
        lg = self.last_logits(prompts, batch)[:, letter_ids]
        return torch.log_softmax(lg, -1).numpy()

    # ------------------------------------------------------------------ hooks
    @contextlib.contextmanager
    def hooked(self, fn_by_hidden_index):
        """fn_by_hidden_index: {hidden index j (1..n_layers): fn(h)->h}. Applied to block j-1."""
        handles = [self.layers[j - 1].register_forward_hook(_wrap(fn))
                   for j, fn in fn_by_hidden_index.items()]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def to_dev(self, arr):
        return torch.as_tensor(np.asarray(arr), device=self.device, dtype=self.dtype)
