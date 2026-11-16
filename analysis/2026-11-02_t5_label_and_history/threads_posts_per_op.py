"""All OPs of threads.jsonl: posts per OP, and for every post the number of earlier posts by the same OP (inside threads.jsonl).

Usage: python threads_posts_per_op.py <cmv_dir>   (<cmv_dir> holds the Webis-CMV-20 files; no other arguments)
"""
import sys
import bz2, json, datetime
from collections import defaultdict, Counter
import numpy as np
D = sys.argv[1]  # folder with the Webis-CMV-20 .jsonl.bz2 files
by=defaultdict(list); deleted=0; n=0
with bz2.open(f"{D}/threads.jsonl.bz2","rt") as h:
    for line in h:
        r=json.loads(line); n+=1
        a=r.get("author")
        if a in (None,"[deleted]"): deleted+=1; continue
        by[a].append(int(r["created_utc"]))
per_op=np.array([len(v) for v in by.values()])
earlier=[]; year_first=Counter(); year_all=Counter()
for a,ts in by.items():
    ts.sort()
    for i,t in enumerate(ts):
        earlier.append(i)   # i posts of the same OP are strictly earlier (ties are rare)
        y=datetime.datetime.fromtimestamp(t,datetime.timezone.utc).year; year_all[y]+=1; year_first[y]+=(i==0)
e=np.array(earlier)
def pct(v): return " ".join(f"{k} {int(np.percentile(v,p))}" for k,p in (("p25",25),("median",50),("p75",75),("p90",90),("p99",99)))+f" max {int(v.max())}"
print(f"threads.jsonl: {n} posts | deleted/unknown author: {deleted} | posts with a known OP: {len(e)} | distinct OPs: {len(by)}")
print("\nPOSTS PER OP (all of threads.jsonl):", pct(per_op), f"| mean {per_op.mean():.2f}")
for lo,hi,lab in ((1,1,"1"),(2,2,"2"),(3,3,"3"),(4,5,"4-5"),(6,10,"6-10"),(11,20,"11-20"),(21,10**6,"21+")):
    m=(per_op>=lo)&(per_op<=hi); print(f"   OPs with {lab:>5s} posts: {m.sum():6d} ({m.mean():5.1%} of OPs) -> {per_op[m].sum():6d} posts ({per_op[m].sum()/per_op.sum():5.1%} of posts)")
print("\nEARLIER POSTS OF THE SAME OP, per post:", pct(e), f"| mean {e.mean():.2f}")
for lo,hi,lab in ((0,0,"0"),(1,1,"1"),(2,2,"2"),(3,4,"3-4"),(5,9,"5-9"),(10,19,"10-19"),(20,10**6,"20+")):
    m=(e>=lo)&(e<=hi); print(f"   {lab:>5s} earlier posts: {m.sum():6d} posts ({m.mean():5.1%})")
print("\nshare of posts that are the OP's first post in the file, by year (2013 is high partly because the file starts there):")
print("  ", {y: f"{year_first[y]/year_all[y]:.1%} of {year_all[y]}" for y in sorted(year_all)})
