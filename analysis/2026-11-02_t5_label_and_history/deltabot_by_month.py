"""Per month: delta flag (flair) vs 'Confirmed: ... delta awarded' DeltaBot replies under a comment written by the OP.

Usage: python deltabot_by_month.py <cmv_dir>   (<cmv_dir> holds the Webis-CMV-20 files; no other arguments)
"""
import sys
import bz2, json, datetime
from collections import defaultdict
D = sys.argv[1]  # folder with the Webis-CMV-20 .jsonl.bz2 files
def walk(cs, parent=None):
    for c in cs or []:
        yield c, parent
        yield from walk(c.get("children"), c)
m=defaultdict(lambda:[0,0,0,0,0])  # n, flag, confirmed, union, flag&confirmed
with bz2.open(f"{D}/posts_malleability.jsonl.bz2","rt") as h:
    for line in h:
        r=json.loads(line); op=r.get("author")
        t=datetime.datetime.fromtimestamp(int(r["created_utc"]),datetime.timezone.utc); k=f"{t.year}-{t.month:02d}"
        conf=any(c.get("author")=="DeltaBot" and (c.get("body") or "").startswith("Confirmed") and p is not None and p.get("author")==op for c,p in walk(r.get("comments")))
        flag=bool(r.get("delta")); a=m[k]
        a[0]+=1; a[1]+=flag; a[2]+=conf; a[3]+=(flag or conf); a[4]+=(flag and conf)
json.dump(m,open("bot_label.json","w"),indent=1)
print("month n flag% confirmed% union% flag&confirmed")
for k in sorted(m):
    n,f,c,u,fc=m[k]; print(k,n,round(100*f/n),round(100*c/n),round(100*u/n),fc)
