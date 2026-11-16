"""For every post of the T5 dataset: how many posts did the same OP write earlier in CMV (all threads.jsonl submissions, strictly before)? Positive rate by that count.

Usage: python earlier_posts_per_post.py <cmv_dir>   (<cmv_dir> holds the Webis-CMV-20 files; no other arguments)
"""
import sys
import bz2, json
from collections import defaultdict, Counter
import numpy as np
D = sys.argv[1]  # folder with the Webis-CMV-20 .jsonl.bz2 files
from datasets import load_dataset
tables = load_dataset("nlp-final-project11/persuasion-datasets", "cmv_malleability_liwc")  # needs a Hugging Face login
rows = [{"author": a, "created_utc": t, "label": y} for table in tables.values() for a, t, y in zip(table["author"], table["created_utc"], table["label"])]
ops={r["author"] for r in rows}
times=defaultdict(list)
with bz2.open(f"{D}/threads.jsonl.bz2","rt") as h:
    for line in h:
        r=json.loads(line)
        if r.get("author") in ops: times[r["author"]].append(int(r["created_utc"]))
n=np.array([sum(1 for t in times[r["author"]] if t<r["created_utc"]) for r in rows]); y=np.array([r["label"] for r in rows])
print("T5 posts:",len(rows),"| OPs:",len(ops))
print("earlier own posts per post: mean %.2f median %d p75 %d p90 %d max %d"%(n.mean(),np.median(n),np.percentile(n,75),np.percentile(n,90),n.max()))
buckets=[(0,0),(1,1),(2,2),(3,4),(5,9),(10,10**6)]
print(f"{'earlier own posts':>18s} {'posts':>6s} {'share':>6s} {'delta rate':>10s}")
for lo,hi in buckets:
    m=(n>=lo)&(n<=hi); print(f"{(str(lo) if lo==hi else (str(lo)+'-'+str(hi) if hi<10**6 else str(lo)+'+')):>18s} {m.sum():6d} {m.mean():6.1%} {y[m].mean():10.3f}")
tot=Counter(Counter(r["author"] for r in rows).values())
print("\nposts per OP in the T5 dataset (OPs with k posts):",dict(sorted(tot.items())))
