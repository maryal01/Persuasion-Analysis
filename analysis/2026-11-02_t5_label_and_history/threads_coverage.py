"""How complete is threads.jsonl? Overlap of submission ids with posts_malleability and pairs, years, and found comments vs Reddit's num_comments.

Usage: python threads_coverage.py <cmv_dir>   (<cmv_dir> holds the Webis-CMV-20 files; no other arguments)
"""
import sys
import bz2, json, datetime, statistics
from collections import Counter
D = sys.argv[1]  # folder with the Webis-CMV-20 .jsonl.bz2 files
def ids_of(f, key):
    s=set()
    with bz2.open(f"{D}/{f}","rt") as h:
        for l in h: s.add(key(json.loads(l)))
    return s
pm=ids_of("posts_malleability.jsonl.bz2", lambda r: r["id"])
pairs=ids_of("pairs.jsonl.bz2", lambda r: "t3_"+r["submission_id"] if not r["submission_id"].startswith("t3_") else r["submission_id"])
def flat(cs):
    n=0
    for c in cs or []:
        n+=1+flat(c.get("children"))
    return n
tid=set(); years=Counter(); ratios=[]; nc_zero=0; per_year_ratio={}
with bz2.open(f"{D}/threads.jsonl.bz2","rt") as h:
    for l in h:
        r=json.loads(l); tid.add(r["id"])
        y=datetime.datetime.fromtimestamp(int(r["created_utc"]),datetime.timezone.utc).year; years[y]+=1
        nc=r.get("num_comments") or 0
        if nc>0:
            q=min(flat(r.get("comments"))/nc,1.5); ratios.append(q); per_year_ratio.setdefault(y,[]).append(q)
        else: nc_zero+=1
print("threads.jsonl submissions:",len(tid),"| by year:",dict(sorted(years.items())))
print("posts_malleability ids:",len(pm),"| also in threads:",len(pm&tid),f"({len(pm&tid)/len(pm):.1%})")
print("pairs submission ids:",len(pairs),"| also in threads:",len(pairs&tid),f"({len(pairs&tid)/len(pairs):.1%}) | in posts_malleability:",len(pairs&pm))
print("threads with num_comments=0:",nc_zero)
rs=sorted(ratios); q=lambda p: rs[int(p*(len(rs)-1))]
print(f"comments found / num_comments: p10 {q(.1):.2f} p25 {q(.25):.2f} median {q(.5):.2f} p75 {q(.75):.2f}; share with >=90% found: {sum(1 for x in ratios if x>=0.9)/len(ratios):.1%}")
print("median ratio by year:",{y:round(statistics.median(v),2) for y,v in sorted(per_year_ratio.items())})
