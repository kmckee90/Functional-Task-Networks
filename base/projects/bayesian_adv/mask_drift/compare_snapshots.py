"""Compare exact frozen mask drift for baseline and changepoint storage runs."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "artifacts_full"
RUNS = [
    ("standard", ""),
    ("replace", "_cp"),
    ("append 8", "_cp_append8"),
    ("append 32", "_cp_append32"),
]


def summarize(path, source):
    with (path / "trajectories.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(int(r["block"]), int(r["task"]), r["mask_source"]): r
               for r in rows}
    pairs = [(indexed[(task, task, source)], indexed[(5, task, source)])
             for task in range(5)]

    def avg(values):
        return sum(values) / len(values)

    return {
        "origin_acc": avg([float(a["acc"]) for a, _ in pairs]),
        "final_acc": avg([float(b["acc"]) for _, b in pairs]),
        "delta_ce": avg([float(b["ce"]) - float(a["ce"]) for a, b in pairs]),
        "kl": avg([float(b["kl_from_origin"]) for _, b in pairs]),
    }


def main():
    lines = [
        "# Changepoint storage comparison",
        "",
        "Pure DomainShift, BayesLoRA, standard training schedule and one filtered",
        "replay batch per step. Each accuracy is the mean over old tasks 0–4 on",
        "512 fixed query examples per task. The same exact hard mask is applied at",
        "task completion and after task 5; no evaluation mask search is used.",
        "`replay_best` is selected at task completion from masks then in the",
        "buffer; later changepoint storage may overwrite or evict that entry.",
        "",
        "## Two-seed averages",
        "",
        "| Storage | Mask | Task-end acc | Final acc | Δ acc |",
        "|---|---|---:|---:|---:|",
    ]
    for label, suffix in RUNS:
        for source in ("last_train", "replay_best"):
            results = []
            for seed in (123, 124):
                path = ROOT / f"BayesLoRA_domain_{seed}{suffix}"
                if (path / "train_metrics.json").exists():
                    results.append(summarize(path, source))
            if len(results) == 2:
                origin = sum(d["origin_acc"] for d in results) / 2
                final = sum(d["final_acc"] for d in results) / 2
                lines.append(f"| {label} | {source} | {origin:.3f} | "
                             f"{final:.3f} | {final - origin:+.3f} |")
    lines += [
        "",
        "## Individual runs",
        "",
        "| Seed | Storage | Mask | Task-end acc | Final acc | Δ acc | Δ CE | KL | Events | Final buffer by task |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in (123, 124):
        for label, suffix in RUNS:
            path = ROOT / f"BayesLoRA_domain_{seed}{suffix}"
            if not (path / "trajectories.csv").exists():
                continue
            if not (path / "train_metrics.json").exists():
                continue
            events_file = path / "snapshot_events.json"
            events = json.loads(events_file.read_text()) if events_file.exists() else []
            composition_file = path / "buffer_composition.json"
            composition = json.loads(composition_file.read_text()) if composition_file.exists() else {}
            composition_text = ", ".join(f"{k}:{v}" for k, v in composition.items()) or "—"
            for source in ("last_train", "replay_best"):
                d = summarize(path, source)
                lines.append(
                    f"| {seed} | {label} | {source} | {d['origin_acc']:.3f} | "
                    f"{d['final_acc']:.3f} | {d['final_acc'] - d['origin_acc']:+.3f} | "
                    f"{d['delta_ce']:+.3f} | {d['kl']:.3f} | {len(events)} | "
                    f"{composition_text} |"
                )
    lines += [
        "",
        "`replace` refreshes existing buffer slots with recent outgoing-mask data;",
        "`append` adds recent outgoing-mask data to the same fixed-capacity deque,",
        "evicting its oldest entries. Neither policy changes the replay overlap",
        "filter or the number of replay gradient batches per step.",
        "",
    ]
    lines += [
        "## Oldest task (task 0)",
        "",
        "| Seed | Storage | Mask | Task-end acc | Final acc | Final buffer entries |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for seed in (123, 124):
        for label, suffix in RUNS:
            path = ROOT / f"BayesLoRA_domain_{seed}{suffix}"
            if not (path / "train_metrics.json").exists():
                continue
            with (path / "trajectories.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            by_key = {(int(r["block"]), int(r["task"]), r["mask_source"]): r
                      for r in rows}
            composition_file = path / "buffer_composition.json"
            composition = json.loads(composition_file.read_text()) if composition_file.exists() else {}
            n = composition.get("0", "—")
            for source in ("last_train", "replay_best"):
                origin = float(by_key[(0, 0, source)]["acc"])
                final = float(by_key[(5, 0, source)]["acc"])
                lines.append(f"| {seed} | {label} | {source} | {origin:.3f} | {final:.3f} | {n} |")
    lines.append("")
    output = Path(__file__).resolve().parent / "SNAPSHOT_REPORT.md"
    output.write_text("\n".join(lines))
    print(output.read_text())


if __name__ == "__main__":
    main()
