"""Pod-only diagnosis and repair of a saved input-conditioned mask memory."""

import argparse
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
import config
import run_experiments as experiment
from routing import BayesLoRAFTN, BayesLoRARNNFTN, ReconstructionPrior

# The runner is executed as __main__, so its replay class has that pickle path.
ReplayBuffer = experiment.ReplayBuffer


def score(model, x, y, mask):
    with torch.no_grad():
        pred = model(x, m=mask)
        return experiment.accuracy(pred, y), model.loss(pred, y).item()


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--fit-rows", type=int, default=512)
parser.add_argument("--steps", type=int, default=30)
args = parser.parse_args()

torch.set_default_device("cuda")
saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
bases = tuple(getattr(config, name) for name in reversed(saved["config_names"]))
cfg = type("OfflineConfig", bases, {})()
experiment.Config = cfg
dg = experiment.make_data_gen(saved["seed"] + 100)
arch = cfg.ARCHITECTURE
cls = BayesLoRAFTN if arch == "bayes_lora" else BayesLoRARNNFTN
pars = (dict(cfg.LORA_FTN_PARS) if arch == "bayes_lora"
        else dict(cfg.LORA_RNN_FTN_PARS))
model = cls(input_size=dg.total_input_dim, output_size=dg.output_dim,
            hidden_size=cfg.HIDDEN_SIZE, **pars, **cfg.BAYES_FTN_PARS)
model.configure_ic(cfg)
model.load_state_dict(saved["model"])
model.ic_entries = saved["ic_entries"]
model.ic_cov = saved["ic_cov"]
model.ic_samples = saved["ic_samples"]
model.ic_support_x = saved["ic_support_x"]
model.ic_support_y = saved["ic_support_y"]
model.eval()

xs, _ = dg(args.task)
entry, error = model._ic_match(xs)
assert entry is not None, f"no route match for task {args.task}: {error}"
rows = []
for xr, yr in saved["replay"].data:
    matched, _ = model._ic_match(xr)
    if matched is not None and torch.linalg.norm(
            matched.projector - entry.projector).item() < 0.05:
        rows.append((xr, yr))
assert rows, "replay has no matching input domain"
x = torch.cat([v[0] for v in rows])
y = torch.cat([v[1] for v in rows])
order = torch.randperm(x.shape[0])
x, y = x[order], y[order]
n_fit = min(args.fit_rows, x.shape[0] // 2)
xf, yf = x[:n_fit], y[:n_fit]
xv, yv = x[n_fit:], y[n_fit:]
q = [dg(args.task) for _ in range(8)]
xq = torch.cat([v[0] for v in q])
yq = torch.cat([v[1] for v in q])
print(f"task={args.task} recon_error={error:.3g} replay_rows={len(x)} "
      f"fit={len(xf)} validation={len(xv)} fresh={len(xq)}", flush=True)

candidates = []
initial = model._topk_mask(entry.logits)
candidates.append(("stored", initial.detach().clone()))
state = model.get_reconfig_state()

for kl in (0.0, 0.01, 0.1):
    model.mask_logits.copy_(entry.logits)
    model._posterior_update(xf, yf, steps=args.steps, lr=0.3,
                            sample_size=None, kl_weight=kl, temperature=1.0,
                            prior_decay=1.0, stochastic_mask=False,
                            changepoint=False)
    candidates.append((f"warm_kl{kl}", model._topk_mask(model.mask_logits).clone()))
    model.set_reconfig_state(state)

for restart in (1, 3):
    super(ReconstructionPrior, model).cold_reconfigure(
        xf, yf, steps=args.steps, lr=0.3, sample_size=None,
        temperature=1.0, restarts=restart, take_best=True)
    candidates.append((f"cold_{restart}", model.m.clone()))
    model.set_reconfig_state(state)

best = None
for name, mask in candidates:
    fit_acc, fit_loss = score(model, xf, yf, mask)
    val_acc, val_loss = score(model, xv, yv, mask)
    fresh_acc, fresh_loss = score(model, xq, yq, mask)
    print(f"{name:12s} fit={fit_acc:.3f}/{fit_loss:.3f} "
          f"val={val_acc:.3f}/{val_loss:.3f} "
          f"fresh={fresh_acc:.3f}/{fresh_loss:.3f}", flush=True)
    if best is None or val_loss < best[0]:
        best = (val_loss, name, fresh_acc)
print(f"best_validation={best[1]} fresh_accuracy={best[2]:.3f}")
