# Bayesian FTN variant of RECODE M2E

Run on the m2e pod, from its workspace:

```bash
uv run python experiment_ftn.py --task Fixed --device cuda --seed 1901
```

`experiment_ftn.py` preserves the `experiment_recode.py` experiment and exposes
FTN controls in its `Hyperparameters` dataclass. `agent_ftn.RDDPG_Dual` inherits
the original agent. Its replay method keeps the original frontier selection,
goal-buffer movement, prioritized contiguous sampling and reservoir encoding;
only the decoder optimization block changes. The original files are untouched.

Unchanged: observation and feedback reservoirs, RECODE density and feedback,
epsilon behavior, action normalization, actor/dual-critic DDPG objectives,
discounts, OCP/FCP ablations, optional empirical goal-return blending (`a`),
replay cadence, and target Polyak update rate.

## FTN decoders

Each of the observation critic, feedback critic and actor has an independent
bank of deep neurons and a Bayesian exact-k mask. Separate masks follow
`experiments/rl/robotics/train_cw.py` and `train_mt10_ddpg.py`; a single DQN
mask cannot directly describe these three different DDPG functions.

Critics select masks by squared TD error against detached bootstrap labels.
Those labels are held fixed for each mask search, then recomputed for training.
The actor searches after the critics, maximizing the original OCP/FCP-weighted
target-critic value through its actions. This uses the original M2E target
critics rather than changing the actor's learning objective. Mask inference
does not train network weights. No task labels or external reward are added.

Globo's Bayesian posterior implementation is vendored under
`agents/ftn_support/`. Default searches take two steps, with persistent logits,
a KL prior, and change-point refitting when the combined critic loss exceeds
its EMA by three standard deviations after 32 searches. Action-time searches
use the latest replay support every eight interactions, once a minibatch is
available. They preserve all four reservoir hidden states; the normal forward
path still follows the original agent's epsilon and reservoir behavior.

Defaults use width 8, depth 8 and eight times as many bank neurons as selected
neurons. If `ftn_k` is None, each decoder's active count is calculated to roughly
match its original decoder's trainable parameter count; integer rounding means
it is not an exact match. `ftn_hidden_size` overrides the total bank size.
`agent.ftn_spec` reports the actual counts for all three decoders.

Every eighth update archives a support and its three masks in a bounded,
uniform-lifetime reservoir (256 entries). One archived support with 0.2–0.8
overlap in at least one mask contributes an additional loss under its saved
masks. Supports retain their sampled reward values; bootstrap targets are
recomputed. They store frozen reservoir features, so replay does not overwrite
the live recurrent context. RMSprop updates only the union of active atoms,
including their optimizer state. Set `ftn_replay_capacity=0` to disable this
auxiliary replay without disabling M2E's original replay/frontier buffers.

Masks and posterior logits are registered buffers. Agent state dictionaries
also retain the FTN support archive and change-point statistics; whole-agent
serialization retains optimizers as in the original experiment. Resetting
memories also clears the FTN support archive. Spatial mask visualization is
not part of this maze experiment.

Tests on the pod:

```bash
OMP_NUM_THREADS=2 uv run python -m unittest test_agent_ftn test_agent_recode -v
```
