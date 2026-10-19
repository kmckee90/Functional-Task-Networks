"""Summarize fixed-mask drift trajectories from the full pod runs."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "artifacts_full"
ARCHES = ("Base", "LoRA", "Bayes", "BayesLoRA")
SHIFTS = ("domain", "concept")
SOURCES = ("replay_best", "last_train", "cold_oracle")


def load(path):
    with path.open(newline="") as f:
        return [{**row, **{k: float(row[k]) for k in (
            "acc", "ce", "kl_from_origin", "tv_from_origin",
            "class_flip_fraction")},
            "block": int(row["block"]), "task": int(row["task"])}
            for row in csv.DictReader(f)]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    out = ["# Exact-mask prediction drift", "",
           "Each row compares fixed query examples under exactly the same task mask",
           "at task completion and after the last task block. The mask is passed",
           "explicitly to the model; no retrospective search occurs. `replay_best`",
           "is chosen on separate support from 16 evenly spaced masks saved in",
           "the ordinary training replay buffer. `last_train` is the final",
           "training mask, and `cold_oracle` is searched once at task completion.",
           "All subsequent evaluations keep those masks fixed. The table pools",
           "tasks 0–4; task 5 has no subsequent training block.", "",
           "| Model | Shift | Mask | Origin acc | Final acc | Δ acc | Δ CE | Mean KL | Mean TV | Flipped classes | Origin overlap with task-end mask |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    details = []
    for arch in ARCHES:
        for shift in SHIFTS:
            path = ROOT / f"{arch}_{shift}_123" / "trajectories.csv"
            if not path.exists() or not (path.parent / "train_metrics.json").exists():
                continue
            rows = load(path)
            end = max(r["block"] for r in rows)
            for source in SOURCES:
                origin = {r["task"]: r for r in rows
                          if r["mask_source"] == source and r["block"] == r["task"]}
                final = {r["task"]: r for r in rows
                         if r["mask_source"] == source and r["block"] == end
                         and r["task"] < end}
                tasks = sorted(final)
                if not tasks:
                    continue
                x = [origin[t] for t in tasks]
                y = [final[t] for t in tasks]
                out.append(
                    f"| {arch} | {shift} | {source} | {mean([r['acc'] for r in x]):.3f} "
                    f"| {mean([r['acc'] for r in y]):.3f} "
                    f"| {mean([y[i]['acc']-x[i]['acc'] for i in range(len(tasks))]):+.3f} "
                    f"| {mean([y[i]['ce']-x[i]['ce'] for i in range(len(tasks))]):+.3f} "
                    f"| {mean([r['kl_from_origin'] for r in y]):.3f} "
                    f"| {mean([r['tv_from_origin'] for r in y]):.3f} "
                    f"| {mean([r['class_flip_fraction'] for r in y]):.3f} "
                    f"| {mean([float(r['overlap_current_mask']) for r in x]):.3f} |")
                if source == "replay_best":
                    details.append((arch, shift, tasks, x, y))
    out.extend(["", "## Stored replay mask by task", "",
                "Accuracy and cross-entropy are evaluated on the identical 512",
                "held-out examples at both times. KL and flip fraction compare",
                "the two predictions on those same examples.", "",
                "| Model | Shift | Task | Origin acc | Final acc | Origin CE | Final CE | KL | Flips |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arch, shift, tasks, origin, final in details:
        for task, a, b in zip(tasks, origin, final):
            out.append(f"| {arch} | {shift} | {task} | {a['acc']:.3f} | {b['acc']:.3f} "
                       f"| {a['ce']:.3f} | {b['ce']:.3f} | {b['kl_from_origin']:.3f} "
                       f"| {b['class_flip_fraction']:.3f} |")
    out.extend(["", "## No replay gradient control", "",
                "The replay buffer still stores its usual data and masks, but",
                "`N_REPLAY_SAMPLES=0` disables replay gradient updates. These are",
                "separate training trajectories at the same seed; compare each",
                "run's own origin and final measurements for a fixed mask.", "",
                "| Model | Shift | Replay batches | Origin acc | Final acc | Δ acc | Δ CE | Mean KL | Flipped classes |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for arch in ("Base", "Bayes"):
        for shift in SHIFTS:
            for replay, tag in ((1, ""), (0, "_r0")):
                path = ROOT / f"{arch}_{shift}_123{tag}" / "trajectories.csv"
                if not path.exists() or not (path.parent / "train_metrics.json").exists():
                    continue
                rows = load(path)
                end = max(r["block"] for r in rows)
                origin = {r["task"]: r for r in rows if r["mask_source"] == "replay_best"
                          and r["block"] == r["task"]}
                final = {r["task"]: r for r in rows if r["mask_source"] == "replay_best"
                         and r["block"] == end and r["task"] < end}
                tasks = sorted(final)
                x = [origin[t] for t in tasks]
                y = [final[t] for t in tasks]
                out.append(f"| {arch} | {shift} | {replay} "
                           f"| {mean([r['acc'] for r in x]):.3f} "
                           f"| {mean([r['acc'] for r in y]):.3f} "
                           f"| {mean([b['acc']-a['acc'] for a,b in zip(x,y)]):+.3f} "
                           f"| {mean([b['ce']-a['ce'] for a,b in zip(x,y)]):+.3f} "
                           f"| {mean([r['kl_from_origin'] for r in y]):.3f} "
                           f"| {mean([r['class_flip_fraction'] for r in y]):.3f} |")
    out.extend(["", "## BayesLoRA DomainShift second seed", "",
                "The task-end mask is a separate exact mask snapshot at the end",
                "of each task block. The training loop only adds masks to the",
                "ordinary replay buffer during the first 100 batches, so it",
                "does not explicitly add this task-end snapshot.", "",
                "| Seed | Mask | Origin acc | Final acc | Δ acc | Δ CE | Mean KL | Origin overlap with task-end mask |",
                "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for seed in (123, 124):
        path = ROOT / f"BayesLoRA_domain_{seed}" / "trajectories.csv"
        if not path.exists() or not (path.parent / "train_metrics.json").exists():
            continue
        rows = load(path)
        end = max(r["block"] for r in rows)
        for source in ("last_train", "replay_best"):
            origin = {r["task"]: r for r in rows if r["mask_source"] == source
                      and r["block"] == r["task"]}
            final = {r["task"]: r for r in rows if r["mask_source"] == source
                     and r["block"] == end and r["task"] < end}
            tasks = sorted(final)
            x = [origin[t] for t in tasks]
            y = [final[t] for t in tasks]
            out.append(f"| {seed} | {source} | {mean([r['acc'] for r in x]):.3f} "
                       f"| {mean([r['acc'] for r in y]):.3f} "
                       f"| {mean([b['acc']-a['acc'] for a,b in zip(x,y)]):+.3f} "
                       f"| {mean([b['ce']-a['ce'] for a,b in zip(x,y)]):+.3f} "
                       f"| {mean([r['kl_from_origin'] for r in y]):.3f} "
                       f"| {mean([float(r['overlap_current_mask']) for r in x]):.3f} |")
    out.extend(["", "## Figures", "",
                "- [BayesLoRA DomainShift task-end masks](artifacts_full/BayesLoRA_domain_123/drift_last_train.png)",
                "- [BayesLoRA DomainShift stored replay masks](artifacts_full/BayesLoRA_domain_123/drift_replay_best.png)",
                "- [Base concept shift with replay](artifacts_full/Base_concept_123/drift_replay_best.png)",
                "- [Base concept shift without replay gradients](artifacts_full/Base_concept_123_r0/drift_replay_best.png)",
                "", "## Interpretation", "",
                "Nonzero KL and class flips with an unchanged hard mask directly",
                "show that changes in predictor weights alter the function",
                "associated with that task mask. Accuracy may improve or decline;",
                "prediction drift alone is not evidence of harmful forgetting.",
                "The task-wise rows identify where the stored mask became worse.",
                "", "With standard replay, Base and Bayes stored masks exhibit",
                "prediction drift, yet most old-task accuracy remains intact.",
                "The no-gradient-replay Base controls lose 12.9 points of",
                "DomainShift accuracy and 20.5 points of concept-shift accuracy",
                "relative to their own task-end baselines. Bayes loses 3.6 and",
                "4.1 points, respectively. These are separate model training",
                "trajectories, not paired weight interventions.",
                "", "BayesLoRA's task-end masks become markedly worse even with",
                "standard replay: 21.8 and 20.3 accuracy points lost in two",
                "DomainShift seeds. The masks sampled from the replay buffer",
                "start weaker and remain more stable. They overlap the task-end",
                "mask by only about one third at the end of training. This is",
                "direct evidence that upkeep of stored early masks does not",
                "guarantee upkeep of a later task-end solution.",
                "", "LoRA and BayesLoRA have substantially larger fixed-mask",
                "prediction drift than Base and Bayes in the seed-123 runs.",
                "This is one seed with the parent's usual filtered replay and",
                "training schedules, except for the second BayesLoRA DomainShift",
                "seed. DomainShift disables concept shift; concept uses the",
                "parent's default task generator settings. Fixed query examples",
                "are generated from a separate RNG stream of the same task",
                "generator; mask selection uses a disjoint support batch.", ""])
    report = Path(__file__).resolve().parent / "REPORT.md"
    report.write_text("\n".join(out))
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
