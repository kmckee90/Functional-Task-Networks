"""Print the executor gate table (design 16.1) from ORACLE eval.json files."""
import glob
import json
import sys

runs = sys.argv[1] if len(sys.argv) > 1 else "/home/dev/gate"
conds = ["id_val", "ood_width_32", "ood_width_64", "ood_width_128",
         "ood_depth_8", "ood_depth_16", "ood_depth_32", "ood_both_64"]
hdr = ["fold"] + ["idv", "w32", "w64", "w128", "d8", "d16", "d32", "both64"]
print("  ".join(f"{h:>7}" for h in hdr) + "   GATE")
allpass = True
for f in sorted(glob.glob(f"{runs}/*/eval/eval.json")):
    ev = json.load(open(f))
    agg = ev["held_out_eval"]["oracle"]
    g = lambda c: agg.get(f"{c}/exact_mean", float("nan"))
    vals = [g(c) for c in conds]
    passed = g("id_val") >= .95 and g("ood_width_64") >= .90 and \
        g("ood_width_128") >= .75 and g("ood_depth_16") >= .75
    allpass = allpass and passed
    print(f"{ev['held_out']:>7}  " + "  ".join(f"{v:7.3f}" for v in vals) +
          f"   {'PASS' if passed else 'FAIL'}")
print("\nExecutor gate 16.1:", "PASS on all evaluated folds" if allpass else "NOT all pass")
print()
for f in sorted(glob.glob(f"{runs}/*/boundary.json")):
    recs = json.load(open(f))
    fr = recs[-1]["r_row"]
    seen = [v for k, v in fr.items() if "/" not in k]
    ho = f.split("heldout-")[1].split("__")[0]
    print(f"{ho:>14}: final mean seen-task exact = {sum(seen)/len(seen):.3f} (n={len(seen)})")
