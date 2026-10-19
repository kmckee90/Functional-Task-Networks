"""Figure 7: controls for the headline result (random-net, support & steps sweeps)."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftn_nar.viz import CAT, BASELINE, INK2, MUTED  # noqa: E402


def main(run="main2"):
    out = Path("results") / run
    c = json.load(open(out / "controls.json"))
    r = json.load(open(out / "results.json"))
    held_reconfig = np.mean([x["reconfig"][0] for x in r["held_rows"]])
    joint_held = np.mean([x["pos"] for x in r["joint_held"]])
    rand_net = np.mean(c["random_net_heldout"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    bars = [("FTN reconfig\n(trained bank)", held_reconfig, CAT[0]),
            ("FTN reconfig\n(RANDOM bank)", rand_net, CAT[2]),
            ("joint task-cond.\nMLP (trained)", joint_held, CAT[3])]
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], 0.55, color=[b[2] for b in bars])
    for x, (_, v, _) in zip(xs, bars):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
    ax.axhline(0.1, color=BASELINE, linewidth=1, linestyle=":")
    ax.text(-0.42, 0.13, "chance", fontsize=8, color=MUTED, ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("per-position accuracy")
    ax.set_title("held-out cells: composition is learned,\nnot configurer expressivity",
                 fontsize=9.5)

    ax = axes[1]
    ns = sorted(int(k) for k in c["support_sweep"])
    means = [np.mean(c["support_sweep"][str(n)]) for n in ns]
    ax.plot(ns, means, marker="o", markersize=5, linewidth=2, color=CAT[0])
    for n, m in zip(ns, means):
        ax.annotate(f"{m:.2f}", (n, m), textcoords="offset points",
                    xytext=(0, 8), fontsize=7.5, color=INK2, ha="center")
    ax.axhline(0.1, color=BASELINE, linewidth=1, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("support-set size (labeled examples)")
    ax.set_ylabel("per-position accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("program induction from few examples", fontsize=9.5)

    ax = axes[2]
    Ss = sorted(int(k) for k in c["steps_sweep"])
    means = [np.mean(c["steps_sweep"][str(S)]) for S in Ss]
    ax.plot(Ss, means, marker="o", markersize=5, linewidth=2, color=CAT[4])
    for S, m in zip(Ss, means):
        ax.annotate(f"{m:.2f}", (S, m), textcoords="offset points",
                    xytext=(0, 8), fontsize=7.5, color=INK2, ha="center")
    ax.axhline(0.1, color=BASELINE, linewidth=1, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ss)
    ax.set_xticklabels(Ss)
    ax.set_xlabel("configurer inner steps S")
    ax.set_ylabel("per-position accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("how much search composition needs", fontsize=9.5)

    fig.savefig(out / "fig7_controls.png", dpi=160, bbox_inches="tight",
                facecolor="#fcfcfb")
    print("fig7 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main2")
