#!/usr/bin/env python
"""Small-model sanity: is 'representation present, behavior absent' real, or do
small models just fail at yes/no instructions in general?

§4.7 claims that sub-3B models encode suicidality strongly (probe ~0.95) yet
read it out at chance behaviorally (AUC ~0.48). A reviewer can object: maybe a
0.5-1B instruct model simply cannot follow ANY yes/no instruction, so the
'behavioral readout absent' result is trivial instruction-following failure, not
a suicidality-specific gap.

Control: run the SAME model, SAME prompt structure, SAME Yes/No readout on a
comparably-simple binary task it SHOULD be able to do behaviorally -- sentiment
(rotten_tomatoes). If the small model gets sentiment behavioral AUC >> 0.5 but
suicide behavioral AUC ~ 0.5, then it CAN act on a yes/no decision in general;
its failure to act on suicidality is specific, and the probe-vs-behavior gap is
real.

Usage:
  python analysis/sanity/sc2_smallmodel_sanity.py --model Qwen/Qwen2.5-0.5B-Instruct
  python analysis/sanity/sc2_smallmodel_sanity.py --model meta-llama/Llama-3.2-1B-Instruct
Output: results/sc2_binary/smallmodel_sanity_{tag}.json
"""
import sys, json, argparse, re
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk, load_dataset
from sklearn.metrics import roc_auc_score
from src.models import ModelLoader

N = 120
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())

SUICIDE_TEMPLATE = (
    "Read the following text and determine if the person shows signs of "
    "suicidal ideation.\n\n"
    "Text: {text}\n\n"
    "Does this text show signs of suicidal ideation? Answer with Yes or No.\n"
    "Answer:"
)
SENTIMENT_TEMPLATE = (
    "Read the following movie review and determine if its sentiment is positive.\n\n"
    "Text: {text}\n\n"
    "Is the sentiment of this review positive? Answer with Yes or No.\n"
    "Answer:"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tag = args.tag or re.sub(r"[^a-zA-Z0-9]+", "_", args.model.split("/")[-1]).lower()

    # suicide task
    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    sui = [t for t, l in zip(ds["text"], ds["label"]) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(ds["text"], ds["label"]) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui, non = sui[:N], non[:N]

    # sentiment task (comparable difficulty yes/no)
    rt = load_dataset("rotten_tomatoes", split="train")
    pos = [r["text"] for r in rt if r["label"] == 1][:N]
    neg = [r["text"] for r in rt if r["label"] == 0][:N]

    loader = ModelLoader(device=device)
    model, tok = loader.load_model(args.model, load_in_4bit=True, auto_quantize=False)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_chat = tok.chat_template is not None
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    print(f"{args.model}: prompt={'chat' if use_chat else 'raw'}")

    def make_ids(text, template):
        prefix, suffix = template.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        if use_chat:
            s = tok.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False)
            return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return tok(content, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def ld(text, template):
        lg = model(make_ids(text, template)).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    def behav_auc(pos_texts, neg_texts, template, name):
        s = []
        for i, t in enumerate(pos_texts + neg_texts):
            s.append(ld(t, template))
            if (i + 1) % 60 == 0:
                print(f"    {name} {i+1}/{len(pos_texts)+len(neg_texts)}")
        y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
        return float(roc_auc_score(y, np.array(s)))

    print("Suicide behavioral AUC...")
    suic_auc = behav_auc(sui, non, SUICIDE_TEMPLATE, "suic")
    print("Sentiment behavioral AUC...")
    sent_auc = behav_auc(pos, neg, SENTIMENT_TEMPLATE, "sent")

    # the model can do yes/no in general if sentiment is clearly above chance
    can_yesno = sent_auc > 0.65
    suicide_specific_gap = can_yesno and suic_auc < 0.60
    print("\n" + "=" * 64)
    print(f"SMALL-MODEL SANITY: {args.model}")
    print("=" * 64)
    print(f"  sentiment behavioral AUC: {sent_auc:.3f}  (can it do yes/no at all?)")
    print(f"  suicide   behavioral AUC: {suic_auc:.3f}")
    if suicide_specific_gap:
        print("VERDICT: SPECIFIC GAP -- model does yes/no (sentiment) but not suicide; "
              "probe-vs-behavior gap is real, not instruction-following failure")
    elif not can_yesno:
        print("VERDICT: model cannot do yes/no in general -- behavioral-absence claim is weak for this model")
    else:
        print("VERDICT: model does both -- not a chance-behavior model")

    ensure = PROJECT_ROOT / "results" / "sc2_binary"
    ensure.mkdir(parents=True, exist_ok=True)
    out = ensure / f"smallmodel_sanity_{tag}.json"
    json.dump({"model": args.model, "tag": tag, "n_per_class": N,
               "prompt": "chat" if use_chat else "raw",
               "sentiment_behavioral_auc": sent_auc,
               "suicide_behavioral_auc": suic_auc,
               "can_do_yesno": bool(can_yesno),
               "suicide_specific_gap": bool(suicide_specific_gap)},
              open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
