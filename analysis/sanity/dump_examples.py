#!/usr/bin/env python
"""Dump worked input -> output examples for the paper.

The result JSONs in results/sc2_binary/ are aggregate (AUC, accuracy, gaps); they
never show what the model was actually fed or what it answered. This script saves
a handful of fully-rendered prompts together with the model's response, so the
paper can show real input/output, not just numbers.

For each sampled post it records:
  - the true label (suicide / non-suicide)
  - the body text (truncated for display)
  - the FULL rendered prompt (chat template, exactly what the model sees)
  - the " Yes" and " No" next-token logits, their difference (the readout used
    everywhere else in the paper), the softmax P(Yes) over {Yes,No}, and the
    argmax prediction + correctness

It also re-runs a couple of examples through all three prompt templates
(standard / empathetic / clinical) to show prompt sensitivity.

Output: results/sc2_binary/examples.json   (+ a printed table)

Run:  python analysis/sanity/dump_examples.py --n 8
Needs the 4-bit Llama on a >=12 GB GPU (same setup as sc2_faithfulness.py).
"""
import sys, json, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

MAXLEN = 256
DISPLAY_CHARS = 600   # how much body text to store for display
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Dump worked input->output examples")
    p.add_argument("--n", type=int, default=8, help="examples per class (default 8)")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui, non = sui[: args.n], non[: args.n]

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]

    def render(text, style="standard"):
        tmpl = PROMPT_TEMPLATES[style]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def run(prompt_str):
        ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        lg = model(ids).logits[0, -1, :].float()
        yl, nl = lg[yes_id].item(), lg[no_id].item()
        p_yes = float(torch.softmax(torch.tensor([yl, nl]), dim=0)[0])
        return {
            "yes_logit": round(yl, 3), "no_logit": round(nl, 3),
            "logit_diff": round(yl - nl, 3), "p_yes": round(p_yes, 4),
            "prediction": "Yes" if yl > nl else "No",
        }

    records = []
    for label, pool in [("suicide", sui), ("non-suicide", non)]:
        for text in pool:
            prompt = render(text, "standard")
            out = run(prompt)
            out["correct"] = (out["prediction"] == "Yes") == (label == "suicide")
            records.append({
                "true_label": label,
                "text_preview": text[:DISPLAY_CHARS] + ("..." if len(text) > DISPLAY_CHARS else ""),
                "text_chars": len(text),
                "rendered_prompt": prompt,
                "template": "standard",
                **out,
            })

    # prompt-sensitivity: first suicide + first non-suicide across all templates
    sensitivity = []
    for label, text in [("suicide", sui[0]), ("non-suicide", non[0])]:
        for style in ("standard", "empathetic", "clinical"):
            out = run(render(text, style))
            sensitivity.append({"true_label": label, "template": style,
                                "text_preview": text[:200] + "...", **out})

    ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
    payload = {
        "dataset": "07_Suicide_Watch_232K",
        "model": "meta-llama/Llama-3.1-8B-Instruct (4-bit NF4)",
        "readout": "logit_diff = logit(' Yes') - logit(' No') at final position",
        "n_per_class": args.n,
        "examples": records,
        "prompt_sensitivity": sensitivity,
    }
    out_path = PROJECT_ROOT / "results" / "sc2_binary" / "examples.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # printed summary table
    print("\n" + "=" * 78)
    print(f"WORKED EXAMPLES (standard template, n={args.n}/class)")
    print("=" * 78)
    print(f"{'true':12} {'pred':5} {'LD':>8} {'P(Yes)':>8}  text")
    print("-" * 78)
    for r in records:
        mark = "" if r["correct"] else "  <-- WRONG"
        snippet = r["text_preview"][:42].replace("\n", " ")
        print(f"{r['true_label']:12} {r['prediction']:5} {r['logit_diff']:8.2f} "
              f"{r['p_yes']:8.3f}  {snippet}{mark}")
    acc = np.mean([r["correct"] for r in records])
    print("-" * 78)
    print(f"argmax accuracy on this sample: {acc:.3f}  "
          f"(expect Yes-bias: high suicide recall, low non-suicide specificity)")
    print(f"\nSaved {out_path}")
    print("Prompt-sensitivity block (2 posts x 3 templates) also saved.")


if __name__ == "__main__":
    main()
