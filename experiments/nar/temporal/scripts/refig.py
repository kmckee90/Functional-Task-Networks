"""Regenerate fig2/fig6 with the id-free re-evaluation overlaid."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftn_temporal.config import full  # noqa: E402
from ftn_temporal.viz import fig_length_ood, fig_methods  # noqa: E402


def main(run="main", suffix="noid"):
    out = Path("results") / run
    results = json.load(open(out / "results.json"))
    reeval = json.load(open(out / f"reeval_v1_{suffix}.json"))
    # keep only the library route (gradient/greedy duplicate the main eval)
    strip = lambda rows: [{k: v for k, v in r.items()
                           if k in ("task", "level", "shortest_word", "library")}
                          for r in rows]
    results["v1_noid"] = dict(trained=strip(reeval["trained"]),
                              held=strip(reeval["held"]))
    cfg = full()
    fig_length_ood(out, results, cfg)
    fig_methods(out, results, cfg)
    print("fig2/fig6 regenerated with id-free overlay")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or []))
