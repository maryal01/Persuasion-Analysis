"""In the T5 dataset: how many label=0 posts have an OP comment with a delta symbol (an attempted award that DeltaBot did not confirm)?

Usage: python label_noise.py <cmv_dir>   (<cmv_dir> holds the Webis-CMV-20 files; no other arguments)
"""
import sys
import bz2, json, re
from collections import Counter
D = sys.argv[1]  # folder with the Webis-CMV-20 .jsonl.bz2 files
from datasets import load_dataset
tables = load_dataset("nlp-final-project11/persuasion-datasets", "cmv_malleability_liwc")  # needs a Hugging Face login
ids = {"t3_" + post_id: label for table in tables.values() for post_id, label in zip(table["post_id"], table["label"])}
sym=re.compile(r"[∆Δ]|&amp;#8710;|!delta",re.I)
def walk(cs,p=None):
    for c in cs or []:
        yield c,p; yield from walk(c.get("children"),c)
cnt=Counter()
with bz2.open(f"{D}/posts_malleability.jsonl.bz2","rt") as h:
    for line in h:
        r=json.loads(line)
        if r["id"] not in ids: continue
        op=r["author"]; label=ids[r["id"]]
        op_sym=False; rejected=False; confirmed=False; bot_any=False
        for c,p in walk(r.get("comments")):
            if p is not None and p.get("author")==op and c.get("author")=="DeltaBot":
                b=c.get("body") or ""; bot_any=True
                confirmed|=b.startswith("Confirmed"); rejected|=("rejected" in b or "disallowed" in b)
            if c.get("author")==op and sym.search(c.get("body") or ""): op_sym=True
        cnt[(label,op_sym,rejected,confirmed)]+=1
tot=Counter()
for (label,op_sym,rej,conf),v in cnt.items(): tot[label]+=v
print("label totals:",dict(tot))
for k,v in sorted(cnt.items()): print("label=%d OP_symbol=%-5s DeltaBot_rejected=%-5s DeltaBot_confirmed=%-5s: %d"%(k+(v,)))
