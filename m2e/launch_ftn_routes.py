"""Launch four independent full FTN runs on the m2e 4x pod, one per GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import mlflow
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_name", default="RDDPG Refinements")
    parser.add_argument("--first_seed", type=int, default=1901)
    parser.add_argument("--no_log", action="store_true", help="Run without Databricks authentication or uploads; keep diagnostics on the pod")
    parser.add_argument("--compare_reactive_low_no_replay", action="store_true", help="Two Current runs and two Reactive_Low NoReplay runs, using matched seeds")
    args = parser.parse_args()
    if torch.cuda.device_count() != 4:
        raise RuntimeError("This launcher requires the m2e 4x pod with four visible GPUs")
    source = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = source / "outputs" / "ftn_route_runs" / stamp
    out.mkdir(parents=True, exist_ok=False)
    os.environ["PATH"] = str(Path.home() / ".local/bin") + os.pathsep + os.environ["PATH"]
    client = experiment = None
    if not args.no_log:
        client = mlflow.MlflowClient(tracking_uri="databricks")
        experiment = client.get_experiment_by_name(f"/Shared/{args.experiment_name}")
        if experiment is None:
            raise RuntimeError("Databricks experiment must exist before launching")
    from experiment_ftn import Hyperparameters, NUM_EPISODES, ftn_setting_overrides
    from dataclasses import asdict
    manifest = dict(launch_time_utc=stamp, experiment_id=experiment.experiment_id if experiment else None,
                    databricks_logging=not args.no_log,
                    comparison="Current vs Reactive_Low NoReplay" if args.compare_reactive_low_no_replay else None,
                    training_episodes=NUM_EPISODES + 1, hyperparameters=asdict(Hyperparameters()),
                    source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [source / "experiment_ftn.py", source / "ftn_diagnostics.py",
                                             source / "ftn_diagnostic_worker.py",
                                             *sorted((source / "agents").rglob("*.py"))]}, runs=[])
    manifest_path = out / "manifest.json"
    for gpu in range(4):
        reactive = args.compare_reactive_low_no_replay and gpu >= 2
        setting = "Reactive_Low" if reactive else "Current"
        condition = "Reactive_Low_NoReplay" if reactive else "Current"
        seed = args.first_seed + (gpu % 2 if args.compare_reactive_low_no_replay else gpu)
        name = f"ftn256_{condition}_seed{seed}_{stamp}"
        resolved = {**asdict(Hyperparameters()), **ftn_setting_overrides(setting, reactive)}
        run = client.create_run(experiment.experiment_id, tags={
            "mlflow.runName": name, "m2e.launch_group": stamp,
            "m2e.gpu": str(gpu), "m2e.purpose": "full FTN mask-route diagnostics",
            "m2e.diagnostics": "async_same_gpu_32_red_png_only", "m2e.condition": condition}) if client else None
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg",
                   SDL_AUDIODRIVER="dummy", PYTHONUNBUFFERED="1")
        log_path = out / f"gpu{gpu}_seed{seed}.log"
        command = [sys.executable, "-u", "experiment_ftn.py", "--device", "cuda", "--task", "Fixed",
                   "--seed", str(seed), "--experiment_name", args.experiment_name,
                   "--run_name", name, "--ftn_setting", setting]
        if run:command.extend(["--log", "--run_id", run.info.run_id])
        if args.no_log:command.extend(["--local_metrics", str(out / f"gpu{gpu}_seed{seed}_metrics.jsonl")])
        if reactive:command.append("--no_ftn_replay")
        try:
            with log_path.open("wb") as log:
                process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            if run:client.set_terminated(run.info.run_id, status="FAILED")
            raise
        record = dict(gpu=gpu, seed=seed, pid=process.pid, name=name, run_id=run.info.run_id if run else None,
                      condition=condition, hyperparameters=resolved, log=str(log_path), command=command)
        manifest["runs"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    print(f"Launch manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
