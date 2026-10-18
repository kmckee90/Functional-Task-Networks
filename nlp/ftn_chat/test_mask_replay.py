"""Replay sampling, overlap filtering, persistence, and mask restoration."""
from io import BytesIO
from types import SimpleNamespace

import torch
from torch import nn

from lora_bank import MaskController
from mask_replay import MaskReplayBuffer
from train import stored_mask_loss


def cfg(**overrides):
    values = dict(H=16, replay_store_prob=1.0, replay_use_prob=1.0,
                  replay_min_gap=3, replay_per_task=2,
                  replay_max_entries=4, replay_overlap_min=0.2,
                  replay_overlap_max=0.8, replay_weight=1.0)
    values.update(overrides)
    return SimpleNamespace(**values)


def example(value):
    ids = torch.tensor([1, 2, value], dtype=torch.long)
    lbl = torch.tensor([-100, -100, value], dtype=torch.long)
    return {"ids": ids, "lbl": lbl, "t_ids": ids.clone(),
            "t_lbl": lbl.clone()}


def mask(*positions):
    out = torch.zeros(16)
    out[list(positions)] = 1
    return out


def test_sparse_reservoir_and_filter():
    gen = torch.Generator(device="cpu").manual_seed(3)
    replay = MaskReplayBuffer(cfg())
    current = mask(0, 1, 2, 3)
    for step in range(10):
        replay.consider("a", [example(3), example(4)], current, gen)
    assert replay.candidates_seen["a"] == 4  # steps 0, 3, 6, 9
    assert len(replay) == 2
    assert all(item["mask"].sum().item() == 4 for item in replay.entries["a"])
    stored_steps = sorted(item["step"] for item in replay.entries["a"])
    assert stored_steps[1] - stored_steps[0] >= 3
    assert replay.sample(current, gen) is None  # exact same mask is excluded

    replay.entries["b"] = [{"mask": mask(0, 1, 8, 9), "example": example(3)}]
    replay.entries["c"] = [{"mask": mask(8, 9, 10, 11), "example": example(4)}]
    assert replay.sample(current, gen)["mask"].equal(replay.entries["b"][0]["mask"])

    blob = BytesIO()
    torch.save(replay.state_dict(), blob)
    blob.seek(0)
    restored = MaskReplayBuffer(cfg())
    restored.load_state_dict(torch.load(blob, weights_only=True))
    assert len(restored) == len(replay)
    assert restored.steps_seen["a"] == 10
    assert restored.sample(current, gen) is not None

    bounded = MaskReplayBuffer(cfg(replay_min_gap=1, replay_max_entries=3))
    for task in ("a", "b", "c"):
        for _ in range(2):
            bounded.consider(task, [example(3)], current, gen)
    assert len(bounded) == 3
    assert all(bounded.entries[task] for task in ("a", "b", "c"))


class TinyLM(nn.Module):
    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.base = nn.Parameter(torch.linspace(-1, 1, 8))
        self.delta = nn.Parameter(torch.linspace(0, 1, 8))

    def forward(self, input_ids, attention_mask=None):
        logits = self.base + self.ctrl.mask.sum() * self.delta
        return SimpleNamespace(logits=logits.expand(input_ids.shape[0],
                                                     input_ids.shape[1], -1))


def test_replay_restores_live_mask():
    ctrl = MaskController(16, 4)
    live = mask(0, 1, 2, 3)
    ctrl.set_mask(live)
    model = TinyLM(ctrl)
    item = {"example": example(3), "mask": mask(0, 1, 8, 9)}
    loss = stored_mask_loss(model, ctrl, item, pad_id=0, dev=torch.device("cpu"))
    assert torch.equal(ctrl.mask, live)
    loss.backward()
    assert model.delta.grad is not None
    assert model.delta.grad.abs().sum().item() > 0


if __name__ == "__main__":
    test_sparse_reservoir_and_filter()
    test_replay_restores_live_mask()
    print("Mask replay checks passed")
