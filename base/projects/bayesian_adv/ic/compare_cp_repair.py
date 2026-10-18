"""Summarize matched BayesLoRA repair × changepoint-replay runs."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = {
    (123, "standard", False): ("artifacts", "sweep_lora_r1_norepair_123"),
    (123, "standard", True): ("artifacts", "lora_full_repair_123"),
    (124, "standard", False): ("artifacts_cp_repair", "standard_norepair_124"),
    (124, "standard", True): ("artifacts", "lora_full_repair_124"),
    **{(seed, "refresh", repair):
       ("artifacts_cp_repair", f"cp_replace_{'repair' if repair else 'norepair'}_{seed}")
       for seed in (123, 124) for repair in (False, True)},
    **{(seed, "priority", repair):
       ("artifacts_cp_repair", f"cp_priority_{'repair' if repair else 'norepair'}_{seed}")
       for seed in (123, 124) for repair in (False, True)},
    **{(seed, "priority-half", repair):
       ("artifacts_cp_repair", f"cp_priority_half_{'repair' if repair else 'norepair'}_{seed}")
       for seed in (123, 124) for repair in (False, True)},
}


def read_run(seed, buffer, repair):
    artifact, name = RUNS[(seed, buffer, repair)]
    metrics_file = ROOT / artifact / name / f"metrics_{name}.txt"
    if not metrics_file.exists():
        return None
    metrics = {}
    for line in metrics_file.read_text().splitlines():
        if " = " in line and not line.startswith("#"):
            key, value = line.split(" = ", 1)
            metrics[key] = value
    log = (ROOT / "artifacts" / "logs" / f"{name}.log"
           if artifact == "artifacts" else ROOT / f"{name}.log")
    final_old = None
    if log.exists():
        matches = re.findall(r"after task 5\).*?acc\[([^\]]+)\]", log.read_text())
        if matches:
            task_accs = [float(v) for _, v in re.findall(r"t(\d+)=([0-9.]+)", matches[-1])]
            final_old = sum(task_accs[:5]) / 5
    return metrics, final_old


def main():
    lines = [
        "# BayesLoRA: does better replay storage reduce IC repair's gain?",
        "",
        "Pure DomainShift, parent-scale six-task run, four input rows for direct",
        "IC routing, one replay-gradient batch per training step.",
        "The refresh policy replaces each outgoing task's existing replay slots",
        "with recent completed batches and the outgoing hard mask at a detected",
        "changepoint. Buffer capacity remains 600 batches; repair does not change",
        "predictor weights or gradient replay. Priority uses the same refreshed",
        "buffer and one replay-gradient batch per step, but samples an old task's",
        "late-mask snapshot even when the ordinary overlap filter excludes it.",
        "Priority-half does so every other step; full priority does so whenever",
        "an old late-mask snapshot is available.",
        "Repair uses 800 fit rows, separate replay validation rows, and three",
        "100-step mask-search restarts after each task block.",
        "",
        "| Seed | Buffer | IC repair | Current-block train acc | Cumulative recall | Final old-task recall | Final IC direct probe | Accepted repairs | Replaced slots | Eligible changes | Priority draws |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    results = {}
    for seed in (123, 124):
        for buffer in ("standard", "refresh", "priority", "priority-half"):
            for repair in (False, True):
                result = read_run(seed, buffer, repair)
                if result is None:
                    continue
                metrics, final_old = result
                results[(seed, buffer, repair)] = metrics
                def fmt(key):
                    value = metrics.get(key, "NA")
                    return f"{float(value):.3f}" if value != "NA" else "—"
                old = f"{final_old:.3f}" if final_old is not None else "—"
                lines.append(
                    f"| {seed} | {buffer} | {'on' if repair else 'off'} | "
                    f"{fmt('avg_train_acc_current_block')} | "
                    f"{fmt('avg_recall_eval_acc')} | {old} | "
                    f"{fmt('ic_probe_direct')} | "
                    f"{metrics.get('ic_repair_accepted', '—')} | "
                    f"{metrics.get('ic_cp_replaced_slots', '—')} | "
                    f"{metrics.get('ic_cp_replay_eligible_events', '—')} | "
                    f"{metrics.get('ic_cp_priority_draws', '—')} |"
                )
    lines += ["", "## Repair gain in cumulative recall", "",
              "| Seed | Standard buffer | Refreshed buffer | Full priority | Half priority |",
              "|---:|---:|---:|---:|---:|"]
    for seed in (123, 124):
        gains = []
        for buffer in ("standard", "refresh", "priority", "priority-half"):
            off = results.get((seed, buffer, False))
            on = results.get((seed, buffer, True))
            gains.append(f"{float(on['avg_recall_eval_acc']) - float(off['avg_recall_eval_acc']):+.3f}"
                         if off and on else "—")
        lines.append(f"| {seed} | {gains[0]} | {gains[1]} | {gains[2]} | {gains[3]} |")
    lines += [
        "",
        "Cumulative recall is the experiment's backward evaluation averaged",
        "over task blocks, including the current task. Final old-task recall",
        "averages tasks 0–4 after task 5. Both use a fresh query batch per task",
        "and direct IC mask retrieval from four unlabeled support inputs.",
        "",
    ]
    output = ROOT / "CP_REPLAY_REPAIR.md"
    output.write_text("\n".join(lines))
    print(output.read_text())


if __name__ == "__main__":
    main()
