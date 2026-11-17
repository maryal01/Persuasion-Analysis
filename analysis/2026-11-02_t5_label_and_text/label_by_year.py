"""Does post['delta'] in posts_malleability mean the OP awarded a delta? Compare with subtree deltas, OP comments carrying a delta symbol, and flair, by year.

Usage: python label_by_year.py <cmv_dir>   (<cmv_dir> holds the Webis-CMV-20 files; no other arguments)
"""
import sys
import bz2, json, re, datetime, sys
from collections import Counter, defaultdict
D = sys.argv[1]  # folder with the Webis-CMV-20 .jsonl.bz2 files
marker=re.compile(r"[∆Δ]|&amp;#8710;|!delta",re.I)
liwc=set()
with bz2.open(f"{D}/author_liwc.jsonl.bz2","rt") as h:
    for l in h: liwc.add(json.loads(l)["author"])
def flat(cs):
    for c in cs or []:
        yield c; yield from flat(c.get("children"))
cross=defaultdict(Counter); flairs=defaultdict(Counter); n=0
with bz2.open(f"{D}/posts_malleability.jsonl.bz2","rt") as h:
    for line in h:
        r=json.loads(line); n+=1
        y=datetime.datetime.fromtimestamp(int(r["created_utc"]),datetime.timezone.utc).year
        sub_delta=any(s.get("delta") for s in r.get("subtrees") or [])
        op_marker=any(c.get("author")==r.get("author") and marker.search(c.get("body") or "") for c in flat(r.get("comments")))
        any_marker=any(marker.search(c.get("body") or "") for c in flat(r.get("comments")))
        key=(bool(r.get("delta")), sub_delta, op_marker)
        cross[y][key]+=1
        flairs[y][(bool(r.get("delta")), (r.get("link_flair_text") or "None")[:25])]+=1
        cross[y][("liwc_op",r.get("author") in liwc)]+=1
        cross[y][("any_delta_symbol_in_thread",any_marker, bool(r.get("delta")))]+=1
out={str(y):{str(k):v for k,v in c.items()} for y,c in cross.items()}
out["flair_top_by_year"]={str(y):{str(k):v for k,v in c.most_common(6)} for y,c in flairs.items()}
json.dump(out,open("label_check.json","w"),indent=1); print("done",n)
