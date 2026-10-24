#!/usr/bin/env python
"""Phase 5, claim 4: Tigges et al. (BlackboxNLP 2024, "Language Models Linearly Represent
Sentiment"; arXiv 2310.15154). Faithful reproduction first, then the F9/F10 controls.

Claim of record (Sec. 4.1, Table 4, Pythia-2.8b):
  SST + suffix " Review Sentiment:", kept items = classified correctly (answer in top 10)
  AND containing >= 1 comma (baseline accuracy 100%).
  4a  mean-ablating the per-layer DAS sentiment direction at every token, every layer:
      logit diff -71%, accuracy -38% (to 62%; chance 50%)
  4b  same with "a small set of random directions": < 1% change
  4c  same direction, comma positions only: logit diff -18%, accuracy -18%
Directional mean ablation (their glossary): r <- r - (r.d) d + mean_i(r_i.d) d.
Directions: DAS (1-d) trained per layer on ToyMovieReview (App. A.7 word lists below),
objective = logit diff after directional patching from the sentiment-flipped prompt.

Part A (reproduce): fit DAS + mean-diff directions per layer; held-out toy patching; 4a/4b/4c.
Part B (controls, as in run_audit.py F9 and steering_mech.py F10):
  - KL-matched necessity: null directions (pca, sign_random, isotropic), made orthogonal
    to the target, mean-ablated with strength beta <= 1 until their KL on neutral
    reference text (AG News, last token) equals the target's; compare SST LD drop.
  - Specificity: the same ablations on an off-target zero-shot task in the same format
    (AG News sports vs business, suffix " Topic:"), filtered to correct items.
  - Comma claim (4c): position-matched null = the same number of random non-comma,
    non-final positions per prompt.

Deviations (recorded in the JSON): our DAS patches the residual stream at all positions of
equal-length toy pairs; ablation hooks act on decoder-block outputs (hidden 1..L), not the
embedding; the mean is the all-token mean of the kept SST prompts; SST = SetFit/sst2 test
sentences (the paper does not name the split for Sec. 4.1); answers are scored as logsumexp over
" positive"/" Positive" (the paper writes the lowercase form in Sec. 4.1, the capitalized one in 3.3).

Usage:
  python repro_tigges.py                                   # Pythia-2.8b, full
  python repro_tigges.py --model EleutherAI/pythia-160m --smoke
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from backend import Backend
from concepts import PROJECT_ROOT
from stats import iut_gate, test_vs_null

# ---------------------------------------------------------------- App. A.7 word lists
POS_ADJ_TRAIN = ("perfect fantastic delightful cheerful good remarkable satisfactory wonderful nice "
                 "fabulous outstanding satisfying awesome exceptional adequate incredible extraordinary "
                 "amazing decent lovely brilliant charming terrific superb spectacular great splendid "
                 "beautiful positive excellent pleasant").split()
NEG_ADJ_TRAIN = ("dreadful bad dull depressing miserable tragic nasty inferior horrific terrible ugly "
                 "disgusting disastrous annoying boring offensive frustrating wretched inadequate dire "
                 "unpleasant horrible disappointing awful").split()
POS_ADJ_TEST = ("stunning impressive admirable phenomenal radiant glorious magical pleasing lively warm "
                "strong helpful vivid modern crisp sweet").split()
NEG_ADJ_TEST = "foul vile appalling rotten grim dismal lazy poor rough noisy sour flat ancient bitter".split()
POS_VERBS = "enjoyed loved liked appreciated admired".split()
NEG_VERBS = "hated disliked despised".split()
POS_ANS = "great amazing awesome good perfect".split()
NEG_ANS = "terrible awful bad horrible disgusting".split()
TOY = "I thought this movie was {adj}, I {verb} it.\nConclusion: This movie is"

PAPER = {"4a_ld_drop": 0.71, "4a_acc_after": 0.62, "4b_change_max": 0.01, "4c_ld_drop": 0.18,
         "4c_acc_drop": 0.18, "imdb_acc_after": 0.57}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-2.8b")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--das-steps", type=int, default=200)
    p.add_argument("--das-batch", type=int, default=16)
    p.add_argument("--das-lr", type=float, default=1e-2)
    p.add_argument("--n-sst", type=int, default=0, help="cap on kept SST items (0 = all)")
    p.add_argument("--n-topic", type=int, default=400, help="AG News sports+business items scored")
    p.add_argument("--n-ref", type=int, default=128, help="neutral reference prompts for KL")
    p.add_argument("--n-null", type=int, default=10, help="draws per null family")
    p.add_argument("--n-random-4b", type=int, default=5, help="random directions, paper's 4b")
    p.add_argument("--pca-k", type=int, default=8)
    p.add_argument("--alpha-level", type=float, default=0.01)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "repro" / "tigges"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.das_steps, a.n_sst, a.n_topic, a.n_ref, a.n_null, a.n_random_4b = 60, 40, 40, 16, 2, 2
    return a


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data
def load_sst(seed):
    from datasets import load_dataset
    ds = load_dataset("SetFit/sst2", split="test")
    items = [(r["text"].strip(), int(r["label"])) for r in ds if r["text"].strip()]
    np.random.default_rng(seed).shuffle(items)
    return items


def load_agnews(seed):
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="test")   # 0 World 1 Sports 2 Business 3 Sci/Tech
    rng = np.random.default_rng(seed)
    rows = [(r["text"].strip(), int(r["label"])) for r in ds]
    rng.shuffle(rows)
    topic = [(t, 1 if l == 1 else 0) for t, l in rows if l in (1, 2)]
    neutral = [t for t, l in rows if l in (0, 3)]
    return topic, neutral


# ---------------------------------------------------------------- forward helpers
class Runner:
    """Batched last-token logits with optional per-layer hooks and per-token position masks."""

    def __init__(self, be, batch):
        self.be, self.batch = be, batch
        self.state = {"mask": None}

    def encode(self, prompts):
        return self.be._encode(prompts)

    @torch.no_grad()
    def logits(self, prompts, hooks=None, masks=None):
        """masks: optional list of per-prompt 0/1 arrays over UNPADDED tokens (1 = ablate here)."""
        out = []
        for i in range(0, len(prompts), self.batch):
            ids, att, pos = self.encode(prompts[i:i + self.batch])
            if masks is not None:
                T = ids.shape[1]
                m = torch.zeros(ids.shape, device=ids.device)
                for b, mk in enumerate(masks[i:i + self.batch]):
                    m[b, T - len(mk):] = torch.as_tensor(mk, dtype=torch.float32, device=ids.device)
                self.state["mask"] = m
            with self.be.hooked(hooks or {}):
                lg = self.be.model(input_ids=ids, attention_mask=att, position_ids=pos,
                                   logits_to_keep=1).logits[:, -1, :]
            out.append(lg.float().cpu())
            self.state["mask"] = None
        return torch.cat(out)

    @torch.no_grad()
    def token_means(self, prompts, n_sample, rng):
        """All-token mean residual per hidden index (non-pad tokens) + a random token sample."""
        L1 = self.be.n_layers + 1
        s, cnt, samp = None, 0, []
        for i in range(0, len(prompts), self.batch):
            ids, att, pos = self.encode(prompts[i:i + self.batch])
            hs = self.be.model(input_ids=ids, attention_mask=att, position_ids=pos,
                               output_hidden_states=True, logits_to_keep=1).hidden_states
            H = torch.stack(hs, 2)[att.bool()].float()          # [ntok, L+1, d]
            s = H.sum(0) if s is None else s + H.sum(0)
            cnt += H.shape[0]
            k = min(len(H), max(1, n_sample * self.batch // max(len(prompts), 1)))
            samp.append(H[torch.as_tensor(rng.choice(len(H), k, replace=False), device=H.device)]
                        .to("cpu", torch.float16))
        assert s is not None and L1 == s.shape[0]
        return (s / cnt).cpu().numpy(), torch.cat(samp).numpy()


def mean_ablation_hooks(be, dirs, mu, beta=1.0, state=None):
    """{j: fn} with r <- r - beta (r.d - mu_j.d) d at hidden index j (block j-1 output).
    If state['mask'] is set ([B,T] float), the edit is applied only where mask = 1."""
    hooks = {}
    for j, d in dirs.items():
        dv = torch.as_tensor(np.asarray(d, dtype=np.float32))
        m0 = float(np.asarray(mu[j], dtype=np.float32) @ np.asarray(d, dtype=np.float32))
        cache = {}

        def fn(h, dv=dv, m0=m0, cache=cache):
            if h.device not in cache:
                cache[h.device] = dv.to(h.device)
            d_ = cache[h.device]
            x = h.float()
            delta = beta * (x @ d_ - m0)
            if state is not None and state["mask"] is not None:
                delta = delta * state["mask"].to(h.device)
            return (x - delta[..., None] * d_).to(h.dtype)
        hooks[j] = fn
    return hooks


def pair_ld(lg, a_ids, b_ids):
    """logsumexp over answer variants (e.g. " positive"/" Positive") of A minus that of B."""
    return (torch.logsumexp(lg[:, a_ids], -1) - torch.logsumexp(lg[:, b_ids], -1)).numpy()


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------------------- toy data / directions
def toy_prompts(adjs_p, adjs_n):
    P = [(TOY.format(adj=a, verb=v), 1, a) for a in adjs_p for v in POS_VERBS]
    N = [(TOY.format(adj=a, verb=v), 0, a) for a in adjs_n for v in NEG_VERBS]
    return P, N


def equal_length_pairs(tok, P, N, rng):
    """(orig, new) prompt pairs of equal token length, both directions (pos->neg, neg->pos)."""
    def bucket(xs):
        b = {}
        for x in xs:
            b.setdefault(len(tok(x[0], add_special_tokens=False)["input_ids"]), []).append(x)
        return b
    bp, bn = bucket(P), bucket(N)
    pairs = []
    for L in set(bp) & set(bn):
        for x in bp[L]:
            pairs.append((x[0], bn[L][rng.integers(len(bn[L]))][0], 0))   # new is negative
        for x in bn[L]:
            pairs.append((x[0], bp[L][rng.integers(len(bp[L]))][0], 1))   # new is positive
    rng.shuffle(pairs)
    return pairs


def answer_ld(lg, pos_ids, neg_ids):
    return lg[:, pos_ids].mean(-1) - lg[:, neg_ids].mean(-1)


def train_das(be, j, pairs, pos_ids, neg_ids, a, rng):
    """1-d DAS at hidden index j: patch the theta-component of the residual (all positions)
    from the flipped prompt; maximize the logit diff toward the flipped prompt's sentiment."""
    d = be.model.config.hidden_size
    g = torch.Generator().manual_seed(a.seed * 1000 + j)
    theta = torch.nn.Parameter((torch.randn(d, generator=g) / d ** 0.5).to(be.device))  # unit-norm init
    opt = torch.optim.Adam([theta], lr=a.das_lr)
    src = {"h": None}

    def fn(h):   # multi-GPU: the hooked layer may sit on a different device than theta/src
        t = (theta / theta.norm()).to(h.device)
        x = h.float()
        return (x + ((src["h"].to(h.device) - x) @ t)[..., None] * t).to(h.dtype)

    sign_tok = lambda y: torch.as_tensor(np.where(np.asarray(y) == 1, 1.0, -1.0), device=be.device)
    pos_t, neg_t = torch.as_tensor(pos_ids, device=be.device), torch.as_tensor(neg_ids, device=be.device)
    for step in range(a.das_steps):
        idx = rng.choice(len(pairs), min(a.das_batch, len(pairs)), replace=False)
        orig = [pairs[i][0] for i in idx]; new = [pairs[i][1] for i in idx]; y = [pairs[i][2] for i in idx]
        ids_n, att_n, pos_n = be._encode(new)
        ids_o, att_o, pos_o = be._encode(orig)
        assert ids_n.shape == ids_o.shape
        with torch.no_grad():
            src["h"] = be.model(input_ids=ids_n, attention_mask=att_n, position_ids=pos_n,
                                output_hidden_states=True, logits_to_keep=1).hidden_states[j].float()
        with be.hooked({j: fn}):
            lg = be.model(input_ids=ids_o, attention_mask=att_o, position_ids=pos_o,
                          logits_to_keep=1).logits[:, -1, :].float()
        ld = lg[:, pos_t].mean(-1) - lg[:, neg_t].mean(-1)
        loss = -(ld * sign_tok(y)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    src["h"] = None
    return unit(theta.detach().float().cpu().numpy())


@torch.no_grad()
def toy_patch_eval(be, j, v, pairs, pos_ids, neg_ids, batch):
    """Directional patching at hidden j along v on held-out pairs: flip rate and mean LD
    change as a fraction of the clean gap (the paper's simple_logit_diff / logit_flip)."""
    flips, frac = [], []
    vt = torch.as_tensor(v, device=be.device, dtype=torch.float32)
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        orig = [c[0] for c in chunk]; new = [c[1] for c in chunk]
        ids_n, att_n, pos_n = be._encode(new); ids_o, att_o, pos_o = be._encode(orig)
        out_n = be.model(input_ids=ids_n, attention_mask=att_n, position_ids=pos_n,
                         output_hidden_states=True, logits_to_keep=1)
        hn = out_n.hidden_states[j].float()
        ld_new = answer_ld(out_n.logits[:, -1, :].float(), pos_ids, neg_ids)
        ld_orig = answer_ld(be.model(input_ids=ids_o, attention_mask=att_o, position_ids=pos_o,
                                     logits_to_keep=1).logits[:, -1, :].float(), pos_ids, neg_ids)
        fn = lambda h: (h.float() + ((hn.to(h.device) - h.float()) @ vt.to(h.device))[..., None]
                        * vt.to(h.device)).to(h.dtype)
        with be.hooked({j: fn}):
            ld_p = answer_ld(be.model(input_ids=ids_o, attention_mask=att_o, position_ids=pos_o,
                                      logits_to_keep=1).logits[:, -1, :].float(), pos_ids, neg_ids)
        flips += (torch.sign(ld_p) != torch.sign(ld_orig)).cpu().tolist()
        frac += ((ld_orig - ld_p) / (ld_orig - ld_new + 1e-6)).cpu().tolist()
    return float(np.mean(flips)), float(np.median(frac))


@torch.no_grad()
def adj_states(be, prompts_adj, batch):
    """Residual at the adjective's last token, all hidden indices: [N, L+1, d] float32."""
    out = []
    for i in range(0, len(prompts_adj), batch):
        chunk = prompts_adj[i:i + batch]
        ids, att, pos = be._encode([c[0] for c in chunk])
        hs = torch.stack(be.model(input_ids=ids, attention_mask=att, position_ids=pos,
                                  output_hidden_states=True, logits_to_keep=1).hidden_states, 1)
        T = ids.shape[1]
        for b, (p, _, adj) in enumerate(chunk):
            n_full = len(be.tok(p, add_special_tokens=False)["input_ids"])
            n_pre = len(be.tok(p.split(f" {adj},")[0] + f" {adj}", add_special_tokens=False)["input_ids"])
            out.append(hs[b, :, T - n_full + n_pre - 1].float().cpu().numpy())
    return np.stack(out)


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^a-z0-9]+", "_", a.model.split("/")[-1].lower()) + f"__seed{a.seed}" + ("_smoke" if a.smoke else "")
    be = Backend(a.model, dtype=a.dtype)
    for prm in be.model.parameters():
        prm.requires_grad_(False)
    L = be.n_layers
    layers = list(range(1, L + 1))
    run = Runner(be, a.batch)
    tid = lambda w: be.token_id(" " + w)
    pos_ids, neg_ids = [tid(w) for w in POS_ANS], [tid(w) for w in NEG_ANS]
    report = {"model": a.model, "seed": a.seed, "args": vars(a), "paper_claims": PAPER,
              "deviations": __doc__.split("Deviations")[1].split("Usage")[0].strip()}
    log(f"repro_tigges on {a.model}: {L} layers")

    # ------------------------------------------------ A1: directions on ToyMovieReview
    Ptr, Ntr = toy_prompts(POS_ADJ_TRAIN, NEG_ADJ_TRAIN)
    Pte, Nte = toy_prompts(POS_ADJ_TEST, NEG_ADJ_TEST)
    tr_pairs = equal_length_pairs(be.tok, Ptr, Ntr, rng)
    te_pairs = equal_length_pairs(be.tok, Pte, Nte, rng)
    Hadj = adj_states(be, Ptr + Ntr, a.batch)
    yadj = np.array([1] * len(Ptr) + [0] * len(Ntr))
    md = {j: unit(Hadj[yadj == 1, j].mean(0) - Hadj[yadj == 0, j].mean(0)) for j in layers}
    das, toy = {}, {}
    for j in (layers[::max(1, L // 4)] if a.smoke else layers):
        das[j] = train_das(be, j, tr_pairs, pos_ids, neg_ids, a, rng)
        if float(das[j] @ md[j]) < 0:
            das[j] = -das[j]                       # orientation: positive = positive sentiment
        fr_d, ch_d = toy_patch_eval(be, j, das[j], te_pairs, pos_ids, neg_ids, a.batch)
        fr_m, ch_m = toy_patch_eval(be, j, md[j], te_pairs, pos_ids, neg_ids, a.batch)
        toy[j] = {"das_flip": fr_d, "das_ld_frac": ch_d, "md_flip": fr_m, "md_ld_frac": ch_m,
                  "cos_das_md": float(das[j] @ md[j])}
        log(f"  L{j:2d} DAS flip {fr_d:.2f} ld {ch_d:+.2f} | MD flip {fr_m:.2f} ld {ch_m:+.2f} | cos {toy[j]['cos_das_md']:+.2f}")
    if a.smoke:   # fill untrained layers with MD so the ablation code path is exercised
        das = {j: das.get(j, md[j]) for j in layers}
    report["toy"] = {"n_train_pairs": len(tr_pairs), "n_test_pairs": len(te_pairs), "per_layer": toy}
    np.savez_compressed(out_dir / f"{tag}_dirs.npz", das=np.stack([das[j] for j in layers]),
                        md=np.stack([md[j] for j in layers]), layers=np.array(layers))

    # ------------------------------------------------ A2: SST subset as in Sec. 4.1
    sst = load_sst(a.seed)
    sst = [(t, y) for t, y in sst if "," in t]
    sp = lambda t: f"{t} Review Sentiment:"
    a_pos, a_neg = [tid("positive"), tid("Positive")], [tid("negative"), tid("Negative")]
    lg = run.logits([sp(t) for t, _ in sst])
    top10 = lg.topk(10, -1).indices
    ld0 = pair_ld(lg, a_pos, a_neg)
    keep = []
    for i, (t, y) in enumerate(sst):
        ans = a_pos if y == 1 else a_neg
        if (ld0[i] > 0) == (y == 1) and bool(torch.isin(top10[i], torch.as_tensor(ans)).any()):
            keep.append(i)
    report["sst_filter"] = {"with_comma": len(sst), "kept": len(keep),
                            "base_acc_with_comma": float(np.mean((ld0 > 0) == (np.array([y for _, y in sst]) == 1)))}
    sst = [sst[i] for i in keep][: (a.n_sst or None)]
    S_prompts = [sp(t) for t, _ in sst]
    S_sign = np.array([1.0 if y == 1 else -1.0 for _, y in sst])
    log(f"SST: {report['sst_filter']['with_comma']} with commas, {len(keep)} kept, using {len(sst)}")
    if not sst:
        raise SystemExit("no SST items survive the correct+top-10 filter; model too weak for the task")

    topic, neutral = load_agnews(a.seed)
    tp = lambda t: f"{t} Topic:"
    t_sp, t_bu = [tid("sports"), tid("Sports")], [tid("business"), tid("Business")]
    topic = topic[: 3 * a.n_topic]
    lgt = run.logits([tp(t) for t, _ in topic])
    ldt = pair_ld(lgt, t_sp, t_bu)
    top10t = lgt.topk(10, -1).indices
    keep_t = [i for i, (t, y) in enumerate(topic)
              if (ldt[i] > 0) == (y == 1) and bool(torch.isin(top10t[i], torch.as_tensor(t_sp if y == 1 else t_bu)).any())]
    topic = [topic[i] for i in keep_t][: a.n_topic]
    T_prompts = [tp(t) for t, _ in topic]
    T_sign = np.array([1.0 if y == 1 else -1.0 for _, y in topic])
    report["topic_filter"] = {"kept": len(topic), "scored": len(keep_t)}
    log(f"AG News topic control: {len(topic)} correct items kept")
    ref_prompts = neutral[: a.n_ref]
    base_lp = torch.log_softmax(run.logits(ref_prompts), -1)

    mu, samp = run.token_means(S_prompts, n_sample=4000, rng=rng)
    def position_masks(prompts):
        """Per-prompt 0/1 masks over unpadded tokens: comma tokens, and the position-matched
        null (the same count of random non-comma, non-final tokens)."""
        commas, rand_pos = [], []
        for p in prompts:
            ids = be.tok(p, add_special_tokens=False)["input_ids"]
            m = np.array(["," in be.tok.decode([t]) for t in ids], dtype=np.float32)
            commas.append(m)
            cand = np.where((m == 0) & (np.arange(len(ids)) < len(ids) - 1))[0]
            r = np.zeros(len(ids), dtype=np.float32)
            r[rng.choice(cand, min(int(m.sum()), len(cand)), replace=False)] = 1
            rand_pos.append(r)
        return {"commas": commas, "randpos": rand_pos}

    S_masks, T_masks = position_masks(S_prompts), position_masks(T_prompts)

    def score(prompts, sign, aid, bid, hooks=None, masks=None):
        if not prompts:
            return float("nan"), float("nan")
        lg_ = run.logits(prompts, hooks, masks)
        ld = pair_ld(lg_, aid, bid) * sign
        return float(ld.mean()), float((ld > 0).mean())

    def kl(hooks):
        lp = torch.log_softmax(run.logits(ref_prompts, hooks), -1)
        return float((base_lp.exp() * (base_lp - lp)).sum(-1).mean())

    S0 = score(S_prompts, S_sign, a_pos, a_neg)
    T0 = score(T_prompts, T_sign, t_sp, t_bu)

    def effect(hooks, where=None):
        """where: None = every token; "commas" | "randpos" = only those positions."""
        s = score(S_prompts, S_sign, a_pos, a_neg, hooks, S_masks[where] if where else None)
        t = score(T_prompts, T_sign, t_sp, t_bu, hooks, T_masks[where] if where else None)
        return {"sst_ld": s[0], "sst_acc": s[1], "sst_ld_drop": 1 - s[0] / S0[0], "sst_acc_drop": S0[1] - s[1],
                "topic_ld": t[0], "topic_acc": t[1], "topic_ld_drop": 1 - t[0] / T0[0],
                "topic_acc_drop": T0[1] - t[1], "kl_ref": kl(hooks) if where is None else None}

    res = {"baseline": {"sst_ld": S0[0], "sst_acc": S0[1], "topic_ld": T0[0], "topic_acc": T0[1]}}
    for name, dirs in (("das", das), ("md", md)):
        res[f"4a_{name}"] = effect(mean_ablation_hooks(be, dirs, mu))
        res[f"4c_{name}_commas"] = effect(mean_ablation_hooks(be, dirs, mu, state=run.state), "commas")
        res[f"4c_{name}_randpos"] = effect(mean_ablation_hooks(be, dirs, mu, state=run.state), "randpos")
        log(f"4a[{name}] SST LD drop {res[f'4a_{name}']['sst_ld_drop']:.1%} acc {res[f'4a_{name}']['sst_acc']:.2f} "
            f"| topic LD drop {res[f'4a_{name}']['topic_ld_drop']:.1%} | KL {res[f'4a_{name}']['kl_ref']:.3f}; "
            f"4c commas LD drop {res[f'4c_{name}_commas']['sst_ld_drop']:.1%} vs rand-pos "
            f"{res[f'4c_{name}_randpos']['sst_ld_drop']:.1%}")
    r4b = [effect(mean_ablation_hooks(be, {j: unit(rng.standard_normal(mu.shape[1])) for j in layers}, mu))
           for _ in range(a.n_random_4b)]
    res["4b_random"] = {"draws": r4b, "sst_ld_drop_mean": float(np.mean([r["sst_ld_drop"] for r in r4b])),
                        "sst_acc_drop_mean": float(np.mean([r["sst_acc_drop"] for r in r4b]))}
    log(f"4b random: SST LD drop {res['4b_random']['sst_ld_drop_mean']:.1%}")
    report["reproduction"] = res

    # ------------------------------------------------ B: KL-matched nulls, orthogonal to target
    tgt = res["4a_das"]
    pcs = {}
    for j in layers:
        X = samp[:, j].astype(np.float32); X = X - X.mean(0)
        pcs[j] = np.linalg.svd(X, full_matrices=False)[2][: a.pca_k]

    def draw(family):
        d = {}
        for j in layers:
            if family == "isotropic":
                v = rng.standard_normal(mu.shape[1])
            elif family == "pca":
                v = rng.standard_normal(a.pca_k) @ pcs[j]
            else:   # sign_random DoM on the toy adjective states
                Dp, Dn = Hadj[yadj == 1, j], Hadj[yadj == 0, j]
                m = min(len(Dp), len(Dn))
                v = rng.choice([-1.0, 1.0], m) @ (Dp[rng.permutation(len(Dp))[:m]] - Dn[rng.permutation(len(Dn))[:m]]) / m
            t_ = das[j]
            d[j] = unit(v - (v @ t_) * t_).astype(np.float32)
        return d

    kl_tests, null_rows = {}, {}
    for fam in ("pca", "sign_random", "isotropic"):
        rows = []
        for _ in range(a.n_null):
            d = draw(fam)
            beta = 1.0
            if kl(mean_ablation_hooks(be, d, mu, 1.0)) > tgt["kl_ref"]:
                lo, hi = 0.0, 1.0
                for _ in range(7):
                    mid = (lo + hi) / 2
                    if kl(mean_ablation_hooks(be, d, mu, mid)) > tgt["kl_ref"]:
                        hi = mid
                    else:
                        lo = mid
                beta = (lo + hi) / 2
            e = effect(mean_ablation_hooks(be, d, mu, beta)); e["beta"] = beta
            rows.append(e)
        null_rows[fam] = rows
        kl_tests[("sst_ld_drop", fam)] = test_vs_null(tgt["sst_ld_drop"], [r["sst_ld_drop"] for r in rows])
        kl_tests[("sst_acc_drop", fam)] = test_vs_null(tgt["sst_acc_drop"], [r["sst_acc_drop"] for r in rows])
        log(f"null[{fam}] KL {np.mean([r['kl_ref'] for r in rows]):.3f} (target {tgt['kl_ref']:.3f}, "
            f"beta {np.mean([r['beta'] for r in rows]):.2f}): SST LD drop "
            f"{np.mean([r['sst_ld_drop'] for r in rows]):.1%}, topic LD drop {np.mean([r['topic_ld_drop'] for r in rows]):.1%}")
    gate = iut_gate(kl_tests, a.alpha_level)
    spec = {"sst_minus_topic_ld_drop": tgt["sst_ld_drop"] - tgt["topic_ld_drop"]}
    report["controls"] = {"kl_matched_necessity": {"target_kl": tgt["kl_ref"], "nulls": null_rows, "gate": gate},
                          "specificity": spec}
    log(f"KL-matched necessity -> {'PASS' if gate['pass'] else 'fail'}; specificity (SST - topic LD drop) "
        f"{spec['sst_minus_topic_ld_drop']:+.1%}")

    report["elapsed_min"] = (time.time() - t0) / 60
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
