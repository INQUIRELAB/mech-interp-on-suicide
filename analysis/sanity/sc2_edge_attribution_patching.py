#!/usr/bin/env python
"""Edge Attribution Patching (EAP) + exact path-patch verification on the binary task.

This is the experiment that adjudicates the word "circuit." The mechanism battery
so far is all NODE-level and converges on "distributed, no critical head":
  - per-head knockout (sc2_head_knockout.py): max AUC drop 0.055, 0/15 survive FDR+CI
  - node AtP (sc2_attribution_patching.py): dense per-head map, no single dominant head
  - activation patching (sc2_patching.py): broad mid-layer IE, no sharp peak
A "circuit" claim, however, is a claim about EDGES -- that a small set of
upstream->downstream connections route the behavior. EAP (Syed, Rager & Conmy
2023, "Attribution Patching Outperforms Automated Circuit Discovery") scores every
edge in the computational graph with ONE forward + ONE backward pass, so we can ask
the edge-level question directly and cheaply.

Graph (residual-stream / layer-component granularity). Nodes:
  - UPSTREAM (write into the residual): embed; each layer's attention output
    (o_proj output); each layer's MLP output (down_proj output).
  - DOWNSTREAM (read from the residual): each layer's attention input
    (input_layernorm input); each layer's MLP input (post_attention_layernorm
    input); the logits node (model.norm input, final position).
An edge u->d exists whenever u writes strictly before d reads.

EAP edge score. Patching edge u->d means: on the corrupt run, replace ONLY the part
of d's input contributed by u with its clean value. Because u writes additively into
the residual stream that d reads, the first-order (Taylor) effect factorizes:

    score(u -> d) ~= ( a_u^clean - a_u^corrupt ) . d(metric)/d(input_d) |_corrupt

i.e. the dot product (over sequence x d_model) of u's clean-minus-corrupt write and
the gradient of the Yes-No logit difference at d's residual read point. We normalize
by (LD_clean - LD_corrupt) so scores are on the same indirect-effect scale as
sc2_patching.json. ONE backward over the corrupt run yields every read-point
gradient at once; one batched matmul against every write difference yields the full
edge matrix.

Exact verification (this is what makes it path patching, not just attribution).
EAP is a linearization. For the top-K edges by |score| we then do the EXACT
single-edge path patch: on the corrupt run, add (a_u^clean - a_u^corrupt) to d's
read point only, and measure the real Yes-No logit difference. We report the
correlation between EAP scores and exact IE for those edges, and use the exact
numbers in the verdict. Single-edge patching costs K forward passes per pair.

Verdict. If a handful of edges carry most of the attribution AND their exact path
patches recover most of the clean-corrupt gap, the routing is a sparse circuit. If
attribution is spread near the uniform-null level across hundreds of edges and no
small edge set is faithful, the routing is a distributed low-rank feature -- which
is what the rest of the battery already indicates, and which would settle the title
on "feature/direction" rather than "circuit."

Output: results/sc2_binary/edge_attribution.json + edge_attribution.png
Run:    python analysis/sanity/sc2_edge_attribution_patching.py --n 30 --verify-topk 12
Cost:   phase 1 ~ (2 fwd + 1 bwd)/pair; phase 2 ~ (verify-topk) fwd/pair. n=30 ~ 10-20 min.
"""
import sys, json, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

MAXLEN = 200          # keep the backward graph inside 12 GB
MIN_LD_GAP = 1.0
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Edge attribution patching (EAP) on binary 232K")
    p.add_argument("--n", type=int, default=30, help="clean/corrupt pairs (default 30)")
    p.add_argument("--offset", type=int, default=0,
                   help="skip this many posts per class in the seeded shuffle; 300 = held out from direction fitting (0-99) and testing (100-299)")
    p.add_argument("--maxlen", type=int, default=MAXLEN)
    p.add_argument("--verify-topk", type=int, default=12,
                   help="exact path-patch this many top edges (0 = skip verification)")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def gini(x):
    """Gini coefficient of a non-negative array; 0 = uniform, ->1 = concentrated."""
    x = np.sort(np.abs(np.asarray(x, dtype=float)))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    maxlen = args.maxlen

    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui, non = sui[args.offset: args.offset + args.n], non[args.offset: args.offset + args.n]

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]

    layers = model.model.layers
    nL = len(layers)
    print(f"{nL} layers; {args.n} pairs, maxlen={maxlen}, verify-topk={args.verify_topk}")

    # ---- node graph: write/read times encode residual-stream causality ----
    # embed writes at -1; attn_L writes at 2L+1, reads at 2L+0.5;
    # mlp_L  writes at 2L+2, reads at 2L+1.5; logits reads at 2*nL+1.
    up_nodes, dn_nodes = [], []
    up_nodes.append(("embed", -1, -1.0))                     # (key, layer, write_time)
    for L in range(nL):
        up_nodes.append(("attn", L, 2 * L + 1.0))
        up_nodes.append(("mlp", L, 2 * L + 2.0))
    for L in range(nL):
        dn_nodes.append(("attn", L, 2 * L + 0.5))            # (key, layer, read_time)
        dn_nodes.append(("mlp", L, 2 * L + 1.5))
    dn_nodes.append(("logits", -1, 2 * nL + 1.0))
    nU, nD = len(up_nodes), len(dn_nodes)
    # causal mask: edge valid iff write_time(u) < read_time(d)
    wt = np.array([w for _, _, w in up_nodes])[:, None]
    rt = np.array([r for _, _, r in dn_nodes])[None, :]
    valid = wt < rt                                          # [nU, nD] bool
    n_edges = int(valid.sum())
    print(f"graph: {nU} upstream x {nD} downstream nodes -> {n_edges} causal edges")

    # ---- hook plumbing -------------------------------------------------------
    cap = {}
    grad_mode = {"on": False}

    def embed_leaf_hook(module, inp, out):
        leaf = out.detach().requires_grad_(True)            # force a grad leaf at the input
        cap[("w", "embed", -1)] = leaf
        return leaf

    def make_write_hook(kind, L):
        def hook(module, inp, out):
            cap[("w", kind, L)] = out[0] if isinstance(out, tuple) else out
        return hook

    def make_read_prehook(kind, L):
        def prehook(module, pargs):
            x = pargs[0]
            cap[("r", kind, L)] = x
            if grad_mode["on"]:
                x.retain_grad()
            return pargs
        return prehook

    handles = [model.model.embed_tokens.register_forward_hook(embed_leaf_hook)]
    for L in range(nL):
        handles.append(layers[L].self_attn.o_proj.register_forward_hook(make_write_hook("attn", L)))
        handles.append(layers[L].mlp.down_proj.register_forward_hook(make_write_hook("mlp", L)))
        handles.append(layers[L].input_layernorm.register_forward_pre_hook(make_read_prehook("attn", L)))
        handles.append(layers[L].post_attention_layernorm.register_forward_pre_hook(make_read_prehook("mlp", L)))
    handles.append(model.model.norm.register_forward_pre_hook(make_read_prehook("logits", -1)))

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: maxlen - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False)["input_ids"]

    def left_pad(a, b):
        L = max(len(a), len(b))
        def pad(x):
            k = L - len(x)
            return [pad_id] * k + x, [0] * k + [1] * len(x)
        ai, am = pad(a); bi, bm = pad(b)
        T = lambda v: torch.tensor([v], device=device)
        return (T(ai), T(am)), (T(bi), T(bm))

    def metric(ids, mask):
        lg = model(input_ids=ids, attention_mask=mask).logits[0, -1, :]
        return lg[yes_id] - lg[no_id]

    def write_val(key, L):
        return cap[("w", key, L)].detach().float()          # [1, seq, d_model]

    # =====================================================================
    # PHASE 1 -- EAP edge scores (one fwd + one bwd per pair)
    # =====================================================================
    edge_sum = np.zeros((nU, nD), dtype=np.float64)
    pairs_cache = []                                         # (cids,cmask,nids,nmask,denom) for phase 2
    n_used = 0

    for pi, (s_text, n_text) in enumerate(zip(sui, non)):
        (cids, cmask), (nids, nmask) = left_pad(chat_ids(s_text), chat_ids(n_text))

        # clean run: cache clean writes (no grad)
        grad_mode["on"] = False
        cap.clear()
        with torch.no_grad():
            ld_clean = metric(cids, cmask).item()
        w_clean = {(k, L): write_val(k, L) for (k, L, _) in up_nodes}

        # corrupt run with grad: cache corrupt writes + read-point grads
        grad_mode["on"] = True
        cap.clear()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            m_corrupt = metric(nids, nmask)
            ld_corrupt = m_corrupt.item()
            denom = ld_clean - ld_corrupt
            if abs(denom) < MIN_LD_GAP:
                print(f"  pair {pi}: |gap|={abs(denom):.2f} < {MIN_LD_GAP}, skipped")
                continue
            m_corrupt.backward()

        w_corrupt = {(k, L): write_val(k, L) for (k, L, _) in up_nodes}

        # stack write-diffs [nU, F] and read-grads [nD, F]; F = seq*d_model
        seq = nids.shape[1]
        dU = torch.stack([(w_clean[(k, L)] - w_corrupt[(k, L)]).reshape(-1)
                          for (k, L, _) in up_nodes])        # [nU, F]
        G = []
        for (k, L, _) in dn_nodes:
            g = cap[("r", k, L)].grad
            G.append((g.float().reshape(-1) if g is not None else torch.zeros_like(dU[0])))
        G = torch.stack(G)                                   # [nD, F]
        edge = (dU @ G.t()).cpu().numpy() / denom            # [nU, nD]
        edge_sum += np.where(valid, edge, 0.0)

        pairs_cache.append((cids, cmask, nids, nmask, denom))
        n_used += 1
        print(f"  pair {pi}: clean LD={ld_clean:+.2f} corrupt LD={ld_corrupt:+.2f} [{n_used} used]")

    if n_used == 0:
        for h in handles:
            h.remove()
        raise RuntimeError("No usable pairs; increase --n.")
    edge_mean = edge_sum / n_used
    edge_mean[~valid] = np.nan

    # ---- rank edges ----
    flat = [(ui, di, float(edge_mean[ui, di]))
            for ui in range(nU) for di in range(nD) if valid[ui, di]]
    flat.sort(key=lambda r: -abs(r[2]))

    def node_str(nodes, i):
        k, L, _ = nodes[i]
        return k if L < 0 else f"{k}{L}"

    def edge_record(ui, di, score):
        return {"src": node_str(up_nodes, ui), "dst": node_str(dn_nodes, di),
                "src_idx": ui, "dst_idx": di, "eap_score": score}

    top_edges = [edge_record(ui, di, s) for ui, di, s in flat[: max(args.verify_topk, 30)]]

    # ---- concentration / sparsity statistics ----
    absvals = np.abs(np.array([s for _, _, s in flat]))
    total = absvals.sum()
    csum = np.cumsum(absvals) / (total + 1e-12)
    def edges_for(frac):
        return int(np.searchsorted(csum, frac) + 1)
    topk_frac = {k: float(csum[min(k, len(csum)) - 1]) for k in (1, 5, 10, 20, 50)}
    uniform_frac = {k: k / n_edges for k in (1, 5, 10, 20, 50)}
    concentration = {
        "n_edges": n_edges,
        "gini": gini(absvals),
        "topk_abs_fraction": topk_frac,
        "uniform_null_fraction": uniform_frac,
        "edges_for_50pct": edges_for(0.50),
        "edges_for_80pct": edges_for(0.80),
        "edges_for_90pct": edges_for(0.90),
    }

    # =====================================================================
    # PHASE 2 -- exact single-edge path patch on the top-K edges
    # =====================================================================
    exact_ie = {}
    if args.verify_topk > 0:
        verify = flat[: args.verify_topk]
        print(f"\nExact path-patching top {len(verify)} edges over {n_used} pairs ...")
        acc = {(ui, di): 0.0 for ui, di, _ in verify}
        for (cids, cmask, nids, nmask, denom) in pairs_cache:
            # cache clean + corrupt writes for this pair (no grad)
            grad_mode["on"] = False
            cap.clear()
            with torch.no_grad():
                _ = metric(cids, cmask)
            w_clean = {(k, L): write_val(k, L) for (k, L, _) in up_nodes}
            cap.clear()
            with torch.no_grad():
                ld_corrupt = metric(nids, nmask).item()
            w_corrupt = {(k, L): write_val(k, L) for (k, L, _) in up_nodes}

            for ui, di, _ in verify:
                uk, uL, _ = up_nodes[ui]
                dk, dL, _ = dn_nodes[di]
                delta = (w_clean[(uk, uL)] - w_corrupt[(uk, uL)])    # [1, seq, d]
                read_mod = (layers[dL].input_layernorm if dk == "attn"
                            else layers[dL].post_attention_layernorm if dk == "mlp"
                            else model.model.norm)
                def add_prehook(module, pargs, delta=delta):
                    return (pargs[0] + delta.to(pargs[0].dtype),) + tuple(pargs[1:])
                h = read_mod.register_forward_pre_hook(add_prehook)
                try:
                    with torch.no_grad():
                        ld_patched = metric(nids, nmask).item()
                finally:
                    h.remove()
                acc[(ui, di)] += (ld_patched - ld_corrupt) / denom
        for (ui, di), v in acc.items():
            exact_ie[(ui, di)] = v / n_used
        for e in top_edges:
            key = (e["src_idx"], e["dst_idx"])
            if key in exact_ie:
                e["exact_ie"] = exact_ie[key]

    for h in handles:
        h.remove()

    # ---- EAP-vs-exact faithfulness + node marginals (validation) ----
    validation = {}
    if exact_ie:
        from scipy.stats import spearmanr, pearsonr
        keys = list(exact_ie.keys())
        eap_v = np.array([edge_mean[ui, di] for ui, di in keys])
        ex_v = np.array([exact_ie[(ui, di)] for ui, di in keys])
        validation["eap_vs_exact_topk"] = {
            "n": len(keys),
            "spearman": float(spearmanr(eap_v, ex_v).statistic),
            "pearson": float(pearsonr(eap_v, ex_v)[0]),
            "exact_ie_sum_topk": float(ex_v.sum()),
        }

    # node marginal strength: total |edge| touching each node
    abs_edge = np.where(valid, np.abs(edge_mean), 0.0)
    up_strength = abs_edge.sum(axis=1)
    dn_strength = abs_edge.sum(axis=0)
    attn_node = np.array([up_strength[i] for i, (k, L, _) in enumerate(up_nodes) if k == "attn"])
    mlp_node = np.array([up_strength[i] for i, (k, L, _) in enumerate(up_nodes) if k == "mlp"])

    def corr(a, b):
        from scipy.stats import spearmanr, pearsonr
        a, b = np.asarray(a, float), np.asarray(b, float)
        return {"spearman": float(spearmanr(a, b).statistic),
                "pearson": float(pearsonr(a, b)[0])}

    pj = PROJECT_ROOT / "results" / "sc2_binary" / "patching.json"
    if pj.exists():
        p = json.load(open(pj))
        if len(p.get("attn_ie", [])) == nL:
            validation["node_vs_patching_attn"] = corr(attn_node, np.abs(p["attn_ie"]))
            validation["node_vs_patching_mlp"] = corr(mlp_node, np.abs(p["mlp_ie"]))
    aj = PROJECT_ROOT / "results" / "sc2_binary" / "attribution_patching.json"
    if aj.exists():
        a = json.load(open(aj))
        if len(a.get("attn_layer_ie", [])) == nL:
            validation["node_vs_atp_attn"] = corr(attn_node, np.abs(a["attn_layer_ie"]))
            validation["node_vs_atp_mlp"] = corr(mlp_node, np.abs(a["mlp_layer_ie"]))

    # ---- verdict ----
    sparse = (concentration["edges_for_80pct"] <= 10
              and topk_frac[5] >= 0.50
              and (validation.get("eap_vs_exact_topk", {}).get("exact_ie_sum_topk", 0.0) >= 0.5))
    verdict = ("SPARSE CIRCUIT: a small edge set concentrates attribution and its exact "
               "path patches recover most of the clean-corrupt gap."
               if sparse else
               "DISTRIBUTED FEATURE: attribution is spread near the uniform-null level "
               "across many edges and no small edge set is faithful -- consistent with the "
               "low-rank / no-critical-head node results; title on 'feature/direction', not 'circuit'.")

    print("\n" + "=" * 70)
    print(f"EDGE ATTRIBUTION PATCHING (EAP, binary 232K, {n_used} pairs)")
    print(f"  {n_edges} causal edges | Gini={concentration['gini']:.3f}")
    print(f"  top-5 |attr| fraction = {topk_frac[5]:.3f}  (uniform null {uniform_frac[5]:.3f})")
    print(f"  edges for 80% of |attr| = {concentration['edges_for_80pct']}")
    print("  top edges by |EAP|:")
    for e in top_edges[:8]:
        ex = f"  exactIE={e['exact_ie']:+.3f}" if "exact_ie" in e else ""
        print(f"    {e['src']:>6s} -> {e['dst']:<7s}  EAP={e['eap_score']:+.4f}{ex}")
    if "eap_vs_exact_topk" in validation:
        v = validation["eap_vs_exact_topk"]
        print(f"  EAP vs exact (top {v['n']}): spearman={v['spearman']:+.3f} "
              f"pearson={v['pearson']:+.3f}; exact IE sum={v['exact_ie_sum_topk']:+.3f}")
    print(f"  VERDICT: {verdict.split(':')[0]}")
    print("=" * 70)

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"
    ensure_dir(out_dir)
    json.dump({
        "dataset": "07_Suicide_Watch_232K", "method": "edge_attribution_patching_EAP",
        "n_pairs": args.n, "n_used": n_used, "n_layers": nL, "maxlen": maxlen,
        "up_nodes": [node_str(up_nodes, i) for i in range(nU)],
        "dn_nodes": [node_str(dn_nodes, i) for i in range(nD)],
        "concentration": concentration, "top_edges": top_edges,
        "attn_node_strength": attn_node.tolist(), "mlp_node_strength": mlp_node.tolist(),
        "validation": validation, "verdict": verdict,
    }, open(out_dir / "edge_attribution.json", "w"), indent=2)

    # ---- figure ----
    # Rendered at the Computational Linguistics print column (5.31 in wide) with serif
    # 8 pt type, so \includegraphics does no scaling and the in-figure text matches the
    # body text height. Panels stacked vertically so each is full column width (large
    # and readable); panel meaning lives in the caption (Top/Middle/Bottom), so the
    # in-figure titles are dropped to keep the panels uncluttered.
    COLW = 5.31
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    npan = 3 if exact_ie else 2
    fig, ax = plt.subplots(npan, 1, figsize=(COLW, 1.9 * npan))

    # (a) edge heatmap
    M = np.where(valid, edge_mean, np.nan)
    vmax = np.nanmax(np.abs(M))
    im = ax[0].imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[0].set_xlabel("downstream node (read)"); ax[0].set_ylabel("upstream node (write)")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046)
    cb.ax.tick_params(labelsize=6)

    # (b) concentration vs uniform null
    ks = np.arange(1, len(csum) + 1)
    ax[1].plot(ks, csum, "-", color="#34558b", lw=1.4, label="EAP cumulative |attr|")
    ax[1].plot(ks, ks / n_edges, "--", color="grey", lw=1.0, label="uniform null")
    ax[1].axhline(0.8, ls=":", c="#b0383d", lw=1)
    ax[1].set_xlabel("edges (ranked by |attribution|)"); ax[1].set_ylabel("cumulative fraction")
    ax[1].set_xlim(0, min(120, n_edges)); ax[1].legend(loc="upper left"); ax[1].grid(alpha=0.25)

    # (c) EAP vs exact path patch
    if exact_ie:
        keys = list(exact_ie.keys())
        eap_v = np.array([edge_mean[ui, di] for ui, di in keys])
        ex_v = np.array([exact_ie[(ui, di)] for ui, di in keys])
        ax[2].scatter(eap_v, ex_v, c="#8e44ad", s=18)
        lim = max(np.abs(eap_v).max(), np.abs(ex_v).max()) * 1.1
        ax[2].plot([-lim, lim], [-lim, lim], "--", c="grey", lw=1)
        ax[2].axhline(0, c="grey", lw=0.5); ax[2].axvline(0, c="grey", lw=0.5)
        ax[2].set_xlabel("EAP score"); ax[2].set_ylabel("exact path-patch IE")
        ax[2].grid(alpha=0.25)

    fig.tight_layout(h_pad=1.0)
    fig.savefig(str(out_dir / "edge_attribution.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/sc2_binary/edge_attribution.json + .png")


if __name__ == "__main__":
    main()
