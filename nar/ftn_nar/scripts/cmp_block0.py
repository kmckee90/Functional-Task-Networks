import glob
import json
import sys

runs = sys.argv[1:] or ["gate2", "lean"]
for tag in runs:
    for f in sorted(glob.glob(f"/home/dev/{tag}/*heldout-MIN_ADD*/boundary.json")):
        recs = [r for r in json.load(open(f)) if r.get("event") == "boundary"]
        r0 = recs[0]
        tr = r0["trained"][-1]
        na = r0["r_row"].get(tr + "/node_acc")
        ex = r0["r_row"].get(tr)
        print(f"{tag}: block0 trained={tr} node_acc={na:.3f} exact={ex:.3f} (n_blocks={len(recs)})")
