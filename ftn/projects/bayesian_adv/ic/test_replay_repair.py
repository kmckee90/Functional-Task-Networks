"""Pod-only check that replay repair preserves the current posterior."""

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from routing import BayesLoRAFTN

torch.set_default_device("cuda")
model = BayesLoRAFTN(input_size=4, output_size=2, hidden_size=16,
                     square_dim=4, n_layers=2, width=16, k=4)
model.configure_ic(SimpleNamespace(
    IC_MODE="reconstruction", IC_TRAIN_PRIOR=False, IC_EVAL_PRIOR=True,
    IC_SUPPORT_SIZE=4, IC_AE_RANK=2, IC_RECON_THRESHOLD=0.1,
    IC_TRIGGER=False, IC_PRIOR_STEPS=2, IC_PRIOR_KL=0.1,
    IC_EVAL_STRATEGY="direct", IC_REFRESH=False, IC_REFRESH_ROWS=16,
    IC_REFRESH_STEPS=2, IC_REFRESH_KL=0.1, IC_REFRESH_METHOD="warm",
    IC_REPLAY_REPAIR=True, IC_REPAIR_FIT_ROWS=64,
    IC_REPAIR_STEPS=2, IC_REPAIR_RESTARTS=1))

replay = SimpleNamespace(data=[])
for domain in range(2):
    for batch in range(16):
        x = torch.zeros(16, 4)
        x[:, 2 * domain:2 * domain + 2] = torch.randn(16, 2)
        y = torch.nn.functional.one_hot(torch.randint(0, 2, (16,)), 2).float()
        replay.data.append((x, y))
        if batch == 0:
            model.reconfigure_changepoint(x, y)
        else:
            model._ic_observe(x, y)

old_logits = model.ic_entries[0].logits.clone()
state = model.get_reconfig_state()
model.ic_repair_from_replay(replay)
assert len(model.ic_repair_before) == 1
assert len(model.ic_repair_after) == 1
assert model.ic_repair_after[0] <= model.ic_repair_before[0]
assert torch.equal(model.ic_entries[0].logits, old_logits)
assert torch.equal(model.mask_logits, state["mask_logits"])
assert torch.equal(model.m, state["m"])
print("replay repair smoke passed")
