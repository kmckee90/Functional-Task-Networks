"""Print paired DomainShift results from this folder's metrics files."""

from pathlib import Path
from statistics import mean


ROOT = Path(__file__).parent / "artifacts"
KEYS = ("avg_recall_eval_acc", "avg_eval_acc_current_task",
        "avg_train_acc_current_block", "ic_train_hits", "ic_eval_hits",
        "ic_cp_holdout_acc", "ic_cp_holdout_loss", "ic_cp_holdout_n")


def read(tag, mode, seed, suffix="paired"):
    name = f"{tag}_{mode}_{seed}_{suffix}"
    path = ROOT / name / f"metrics_{name}.txt"
    values = {}
    for line in path.read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            if key in KEYS:
                values[key] = float(value)
    return values


for tag in ("lora", "rnn", "rnn_io"):
    print(tag)
    for mode in ("base", "prior"):
        rows = [read(tag, mode, seed) for seed in (123, 124, 125)]
        print(mode, "seeds", [round(row["avg_recall_eval_acc"], 4) for row in rows])
        for key in KEYS:
            if all(key in row for row in rows):
                print(" ", key, round(mean(row[key] for row in rows), 4))

for tag in ("lora", "rnn"):
    print(tag, "ablation")
    for mode in ("trainonly", "evalonly"):
        try:
            rows = [read(tag, mode, seed, "ablation")
                    for seed in (123, 124, 125)]
        except FileNotFoundError:
            continue
        print(mode, "seeds", [round(row["avg_recall_eval_acc"], 4) for row in rows])
        for key in KEYS:
            if all(key in row for row in rows):
                print(" ", key, round(mean(row[key] for row in rows), 4))

for tag in ("lora", "rnn"):
    print(tag, "change-point holdout")
    for mode in ("base", "trainonly"):
        rows = [read(tag, mode, seed, "cp") for seed in (123, 124, 125)]
        for key in ("ic_cp_holdout_acc", "ic_cp_holdout_loss", "ic_cp_holdout_n"):
            print(" ", mode, key, round(mean(row[key] for row in rows), 4),
                  "seeds", [round(row[key], 4) for row in rows])
