import json, re, sys, numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score as A
D = sys.argv[1]
def clean(t, n, seed):
    rng = np.random.default_rng(seed); out = [x for x in t if x and len(x) > 80]; rng.shuffle(out); return out[:n]
sui, dep, allrows = [], [], []
for line in open(f"{D}/train.jsonl", encoding="utf-8"):
    r = json.loads(line); lab = r.get("label", ""); txt = (r.get("text") or "").strip()
    allrows.append((lab, txt))
    if lab == "self.SuicideWatch": sui.append(txt)
    elif lab == "self.depression": dep.append(txt)
print("labels:", Counter(l for l, _ in allrows))
pos, neg = clean(sui, 3000, 0), clean(dep, 3000, 1)
fp, fn, ep, en = pos[:2000], neg[:2000], pos[2000:3000], neg[2000:3000]
texts = ep + en; y = np.r_[np.ones(1000), np.zeros(1000)]
tf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), LogisticRegression(C=1.0, max_iter=2000))
tf.fit(fp + fn, [1]*2000 + [0]*2000); s = tf.predict_proba(texts)[:, 1]
z = np.load(f"{D}/suicide_depression__qwen2_5_7b_instruct__seed0.npz")
print("rebuilt tfidf AUC", round(A(y, s), 4), "| saved", round(A(z["y"], z["tfidf"]), 4), "| max abs diff", float(np.abs(s - z["tfidf"]).max()))
norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
# duplicates
sw = set(norm(t) for t in sui); dp = set(norm(t) for t in dep)
print("texts in BOTH SuicideWatch and depression (exact, normalized):", len(sw & dp))
fitset = set(norm(t) for t in fp + fn)
dup_eval = [i for i, t in enumerate(texts) if norm(t) in fitset]
print("eval items with an exact copy in the fit split:", len(dup_eval))
within = Counter(norm(t) for t in texts); print("duplicate texts within eval:", sum(c - 1 for c in within.values() if c > 1))
# keyword presence
kw = re.compile(r"\b(suicid\w*|kill myself|end my life|end it all|want to die|wanna die|take my (own )?life|overdose|hang myself)\b", re.I)
has = np.array([bool(kw.search(t)) for t in texts])
print("explicit suicide keyword present: SuicideWatch %.2f, depression %.2f" % (has[y == 1].mean(), has[y == 0].mean()))
json.dump({"texts": texts, "y": y.tolist(), "tfidf": s.tolist()}, open(f"{D}/eval_rebuilt.json", "w"))
