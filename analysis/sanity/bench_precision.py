#!/usr/bin/env python
"""Timing calibration: how fast is one scored example on this box?

Loads Llama-3.1-8B at the precision given by MI_PRECISION and times the two
shapes of work the sc2_* suite does:

  bs=1  -- one text per forward, hidden states out. This is what every sc2_*
           script does today, so this number sets the wall-clock of a rerun.
  bs=N  -- the same work left-padded into a batch, to size the payoff of
           adding batching before the larger reruns.

Usage:
    MI_PRECISION=fp16 python analysis/sanity/bench_precision.py
    MI_PRECISION=fp16 python analysis/sanity/bench_precision.py --batch 16 32 64
"""
import sys, json, time, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES

MAXLEN = 256
N_WARMUP = 5


def build_texts(n, tok):
    """Filler prompts at the length the real scripts use (~256 tokens)."""
    filler = ("I have been feeling this way for a long time and I do not know "
              "who to talk to about any of it anymore. ")
    body = tok(filler * 40, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    tmpl = PROMPT_TEMPLATES["standard"]
    prefix, suffix = tmpl.split("{text}")
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return [s] * n


@torch.no_grad()
def time_batch(model, tok, prompts, bs, device):
    enc = tok(prompts[:bs], add_special_tokens=False, return_tensors="pt",
              padding=True).to(device)
    for _ in range(N_WARMUP):
        model(**enc, output_hidden_states=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = max(1, 64 // bs)
    for _ in range(reps):
        model(**enc, output_hidden_states=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return bs / dt, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="*", default=[1, 8, 16, 32])
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--gen-tokens", type=int, default=256,
                    help="also time a generation of this many tokens (0 to skip)")
    args = ap.parse_args()

    device = "cuda"
    precision = os.environ.get("MI_PRECISION", "(unset -> caller default)")
    print(f"model={args.model}  MI_PRECISION={precision}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu: {p.name} sm_{p.major}{p.minor} {p.total_memory/1e9:.1f} GB")

    t0 = time.perf_counter()
    loader = ModelLoader(device=device)
    model, tok = loader.load_model(args.model)
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f} s   weights: "
          f"{torch.cuda.memory_allocated()/1e9:.1f} GB allocated")

    prompts = build_texts(max(args.batch), tok)
    out = {"precision": precision, "model": args.model,
           "load_seconds": round(load_s, 1), "forward": {}}

    for bs in args.batch:
        try:
            per_s, dt = time_batch(model, tok, prompts, bs, device)
            out["forward"][bs] = {"seq_per_s": round(per_s, 1),
                                  "batch_seconds": round(dt, 3)}
            print(f"  bs={bs:<3d} {per_s:6.1f} seq/s   "
                  f"({10_000/per_s/60:5.1f} min per 10k forwards)")
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:<3d} OOM")
            torch.cuda.empty_cache()

    if args.gen_tokens:
        enc = tok(prompts[:1], add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=8, do_sample=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(**enc, max_new_tokens=args.gen_tokens, do_sample=False)
            torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        out["generate"] = {"tokens": args.gen_tokens, "seconds": round(gen_s, 2)}
        print(f"  generate {args.gen_tokens} tok, bs=1: {gen_s:.2f} s "
              f"({gen_s*1000/args.gen_tokens:.1f} ms/token)")

    dest = PROJECT_ROOT / "results" / "sc2_binary" / "bench_precision.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    prev[str(precision)] = out
    dest.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
