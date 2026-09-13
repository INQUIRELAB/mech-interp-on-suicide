#!/usr/bin/env python
"""SC-2 positive control: verify the measurement pipeline works.

If Llama can't separate Death-Me from Life-Not-Me (AUC~0.56), is that a
real finding or a broken readout? Test Death-Me vs obviously-neutral text.
  control AUC high (~0.9) -> pipeline works; the NO-GO is a real, specific
      finding (model detects distress, not implicit intention)
  control AUC ~0.5        -> readout is broken; do not trust the NO-GO
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch, numpy as np
from sklearn.metrics import roc_auc_score
from src.analysis_utils import (
    load_model_wrapped, load_dataset_by_category, get_text, PROMPT_TEMPLATES,
)

N = 40
MAXLEN = 256
device = "cuda"

NEUTRAL = [
    "I had a wonderful day at the park with my friends today. The weather was perfect and we played frisbee for hours.",
    "Just finished baking chocolate chip cookies for the bake sale. They turned out great and the kitchen smells amazing.",
    "Our team won the soccer match this weekend! Everyone played really well and we celebrated with pizza afterward.",
    "I'm so excited for my vacation next month. We're going to the beach and I've already started planning all the fun activities.",
    "The new restaurant downtown is fantastic. I tried their pasta and it was the best meal I've had in ages.",
    "My garden is finally blooming. The tomatoes are coming in nicely and the roses look beautiful this year.",
    "Had a productive day at work and finished my project early. Treating myself to a movie tonight.",
    "Learning to play guitar has been so rewarding. I can finally play my favorite song all the way through.",
    "We adopted a puppy last week and he's the sweetest thing. He loves chasing his tail around the living room.",
    "The hiking trip was incredible. We reached the summit just in time to watch the sunrise over the valley.",
    "I just got a promotion at work! All the hard work this year finally paid off and I'm thrilled.",
    "Spent the afternoon reading a great novel in the sun. Sometimes the simple things are the best.",
    "My sister is getting married next spring and I'm going to be the maid of honor. So happy for her.",
    "The farmers market had the freshest produce today. I picked up some berries and homemade bread.",
    "Finally fixed up my bike and went for a long ride along the river. Felt great to be outside.",
    "We had a lovely family dinner tonight. Mom made her famous lasagna and we all laughed for hours.",
    "I planted some herbs on my balcony and they're thriving. Fresh basil for cooking is the best.",
    "The concert last night was amazing. The band played all my favorite songs and the crowd was electric.",
    "Started a new yoga class this morning and I feel so refreshed. Looking forward to going again.",
    "My coffee shop order was on the house today because I'm a regular. Little things like that make my day.",
    "We repainted the living room this weekend and it looks brand new. So happy with how it turned out.",
    "The kids built an incredible sandcastle at the beach. We spent the whole day soaking up the sun.",
    "I finished knitting a scarf for my grandmother. Can't wait to give it to her at the holidays.",
    "Went apple picking today and came home with a huge basket. Apple pie is on the menu this week.",
    "My friend surprised me with tickets to the museum. The new exhibit was absolutely fascinating.",
    "Got a great deal on a used bookshelf and finally organized all my novels. Feeling accomplished.",
    "The morning jog through the park was perfect. Crisp air, birds singing, a great start to the day.",
    "We tried a new board game tonight and it was hilarious. Game nights with friends are the best.",
    "I baked fresh bread for the first time and it actually worked. The whole house smells wonderful.",
    "Spent the day volunteering at the animal shelter. The dogs were so happy to go on walks.",
    "My tomato plants gave us the best harvest yet. Homemade salsa for everyone this weekend.",
    "The road trip was full of great memories. We sang along to music and stopped at every scenic overlook.",
    "I finally learned to make sushi at home and it came out delicious. Dinner was a big success.",
    "We watched the meteor shower from the rooftop last night. Counted over twenty shooting stars.",
    "My painting got accepted into the local art show. I'm nervous but really proud of the work.",
    "Had brunch with old college friends today. It felt like no time had passed at all.",
    "The new bakery around the corner has the flakiest croissants. My weekend treat is sorted.",
    "I organized my closet and donated a bunch of clothes. Feels good to declutter and help others.",
    "We planted a tree in the backyard for Earth Day. Looking forward to watching it grow over the years.",
    "The puzzle we've been working on for weeks is finally complete. A thousand pieces of satisfaction.",
]

cats = load_dataset_by_category(dataset_name="deepsuimind")
death = cats["Death-Me"][:N]
model, tok = load_model_wrapped("llama", device=device)
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]


def chat_ids(text, style="standard"):
    tmpl = PROMPT_TEMPLATES[style]
    prefix, suffix = tmpl.split("{text}")
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
    )
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


def diffs(texts):
    out = []
    for t in texts:
        ids = chat_ids(t)
        with torch.no_grad():
            lg = model(ids)[0, -1, :]
        out.append((lg[yes_id] - lg[no_id]).item())
    return np.array(out)

d_ld = diffs([get_text(x) for x in death])
n_ld = diffs(NEUTRAL[:N])

y_true = np.concatenate([np.ones(len(d_ld)), np.zeros(len(n_ld))])
y_score = np.concatenate([d_ld, n_ld])
auc = roc_auc_score(y_true, y_score)
acc = ((y_score > 0).astype(int) == y_true).mean()

print("=" * 60)
print(f"SC-2 POSITIVE CONTROL: Death-Me vs neutral text (n={N}/class)")
print("=" * 60)
print(f"Death-Me logit-diff:  {d_ld.mean():7.3f} +/- {d_ld.std():.3f}")
print(f"Neutral  logit-diff:  {n_ld.mean():7.3f} +/- {n_ld.std():.3f}")
print(f"ROC-AUC:              {auc:.4f}")
print(f"argmax(Yes,No) acc:   {acc:.4f}")
print("=" * 60)
print("If AUC >> 0.56: pipeline is sound; Death-Me-vs-Life-Not-Me NO-GO is real.")
