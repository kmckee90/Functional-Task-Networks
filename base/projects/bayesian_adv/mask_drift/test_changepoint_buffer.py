"""Pod-only check: repeated changepoints can refresh an existing slot group."""

import torch

from run_drift import ChangepointReplayBuffer


torch.set_default_device("cuda")
buffer = ChangepointReplayBuffer(maxlen=4, batch_size=2, tail_batches=2)
m1 = torch.tensor([1., 0., 1., 0.])
m2 = torch.tensor([0., 1., 0., 1.])
for value in (1., 2.):
    dat = (torch.full((3, 2), value), torch.zeros(3, 2))
    buffer.add(dat, m1)
    buffer.add_completed_batch(dat)
first = buffer.snapshot_before_change(m1)
assert first["replaced_slots"] == 2

# A new segment has recent data but no newly allocated replay slots.
for value in (3., 4.):
    buffer.add_completed_batch((torch.full((3, 2), value), torch.zeros(3, 2)))
second = buffer.snapshot_before_change(m2)
assert second["replaced_slots"] == 2
assert all(torch.equal(mask, m2) for mask in buffer.masks)
assert float(buffer.data[-1][0][0, 0]) == 4.
print("Repeated changepoint refresh passed on", torch.cuda.get_device_name())

rolling = ChangepointReplayBuffer(maxlen=3, batch_size=2, tail_batches=2,
                                  policy="append", append_batches=2)
for value in (1., 2.):
    rolling.add_completed_batch((torch.full((3, 2), value), torch.zeros(3, 2)))
rolling.snapshot_before_change(m1)
assert len(rolling) == 2
rolling.add((torch.full((3, 2), 3.), torch.zeros(3, 2)), m2)
for value in (3., 4.):
    rolling.add_completed_batch((torch.full((3, 2), value), torch.zeros(3, 2)))
event = rolling.snapshot_before_change(m2)
assert event["appended_slots"] == 2 and len(rolling) == 3
assert all(torch.equal(mask, m2) for mask in rolling.masks)
print("Rolling append and eviction passed on", torch.cuda.get_device_name())
