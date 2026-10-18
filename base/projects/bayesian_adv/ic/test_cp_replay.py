"""Pod-only check of changepoint storage and fixed buffer capacity."""

import torch

from run_experiments import ChangepointReplayBuffer


torch.set_default_device("cuda")
buffer = ChangepointReplayBuffer(maxlen=4, batch_size=2, tail_batches=3)
m_early = torch.tensor([1., 0., 1., 0.])
m_late = torch.tensor([0., 1., 0., 1.])
for value in (1., 2., 3.):
    dat = (torch.full((3, 2), value), torch.zeros(3, 2))
    buffer.add(dat, m_early, task=0)
    buffer.observe_completed_batch(dat, task=0)
event = buffer.snapshot_before_change(m_late)
assert event["replaced_slots"] == 3 and len(buffer) == 3
assert all(torch.equal(m, m_late) for m in buffer.masks)
assert [float(dat[0][0, 0]) for dat in buffer.data] == [1., 2., 3.]
assert buffer.entry_tasks == type(buffer.entry_tasks)([0, 0, 0], maxlen=4)

# New task storage reaches capacity; the next snapshot changes only that task.
dat = (torch.full((3, 2), 4.), torch.zeros(3, 2))
buffer.add(dat, m_early, task=1)
buffer.observe_completed_batch(dat, task=1)
buffer.snapshot_before_change(m_early)
assert len(buffer) == 4
assert [float(dat[0][0, 0]) for dat in buffer.data] == [1., 2., 3., 4.]
assert all(torch.equal(buffer.masks[i], m_late) for i in range(3))
print("Changepoint replay refresh passed on", torch.cuda.get_device_name())
