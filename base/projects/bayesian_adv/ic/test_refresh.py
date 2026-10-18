"""Pod-only smoke check for append-only reconstruction-memory refresh."""

from types import SimpleNamespace
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from routing import BayesLoRAFTN


torch.set_default_device("cuda")
model = BayesLoRAFTN(input_size=3, output_size=2, hidden_size=16,
                     square_dim=4, n_layers=2, width=16, k=4)
model.configure_ic(SimpleNamespace(
    IC_MODE="reconstruction", IC_TRAIN_PRIOR=False, IC_EVAL_PRIOR=True,
    IC_SUPPORT_SIZE=4, IC_AE_RANK=2, IC_RECON_THRESHOLD=0.1,
    IC_TRIGGER=False, IC_PRIOR_STEPS=2, IC_PRIOR_KL=0.1,
    IC_REFRESH=True, IC_REFRESH_ROWS=8, IC_REFRESH_STEPS=2,
    IC_REFRESH_KL=0.1, IC_REFRESH_METHOD=sys.argv[1] if len(sys.argv) > 1 else "warm",
    IC_EVAL_STRATEGY="refine", IC_REPLAY_REPAIR=False,
    IC_REPAIR_FIT_ROWS=8, IC_REPAIR_STEPS=2,
    IC_REPAIR_RESTARTS=1))

for _ in range(2):
    x = torch.randn(16, 3)
    y = torch.nn.functional.one_hot(torch.randint(0, 2, (16,)), 2).float()
    model.reconfigure_changepoint(x, y)

old_entry = model.ic_entries[0]
old_logits = old_entry.logits.clone()
state = model.get_reconfig_state()
model.ic_refresh_memory()

assert len(model.ic_entries) > 1
assert torch.equal(old_entry.logits, old_logits)
assert torch.equal(model.mask_logits, state["mask_logits"])
assert torch.equal(model.m, state["m"])
assert len(model.ic_refresh_after) > 0
print("refresh smoke passed")
