"""For the comments of pairs.jsonl: does the comment's subtree in threads.jsonl contain an OP delta symbol / a DeltaBot confirmation?

Usage: python t1_label_evidence.py <cmv_dir>   (<cmv_dir> holds the Webis-CMV-20 files; no other arguments)
"""
import sys
import bz2, json, re
from collections import Counter
D = sys.argv[1]  # folder with the Webis-CMV-20 .jsonl.bz2 files
sym=re.compile(r"[∆Δ]|&amp;#8710;|!delta",re.I)
want={}
with bz2.open(f"{D}/pairs.jsonl.bz2","rt") as h:
    for l in h:
        r=json.loads(l)
        for side in ("delta_comment","nodelta_comment"): want[r[side]["comments"][0]["id"]]=side
def flat(cs):
    for c in cs or []:
        yield c; yield from flat(c.get("children"))
c=Counter()
with bz2.open(f"{D}/threads.jsonl.bz2","rt") as h:
    for line in h:
        r=json.loads(line); op=r.get("author")
        for com in flat(r.get("comments")):
            side=want.get(com.get("id"))
            if not side: continue
            kids=list(flat(com.get("children")))
            c[(side,"found")]+=1
            c[(side,"OP_symbol")]+=any(k.get("author")==op and sym.search(k.get("body") or "") for k in kids)
            c[(side,"bot_confirmed")]+=any(k.get("author")=="DeltaBot" and (k.get("body") or "").startswith("Confirmed") for k in kids)
print("pair comments:",len(want))
for s in ("delta_comment","nodelta_comment"):
    n=c[(s,"found")]; print(s,"found in threads:",n,{k:f"{c[(s,k)]} ({c[(s,k)]/max(n,1):.1%})" for k in ("OP_symbol","bot_confirmed")})
