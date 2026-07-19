"""Deterministic seed derivation.

Every random draw in the project is keyed by a tuple of (run seed, stream name,
indices...). Streams are independent by construction (numpy SeedSequence), so
generation order never matters and any instance can be regenerated in isolation.
"""

from __future__ import annotations

import zlib

import numpy as np

# Distinct stream names used across the project (asserted disjoint where needed).
STREAM_TRAIN = "train"
STREAM_ID_VAL = "id_val"
STREAM_ID_TEST = "id_test"
STREAM_OOD = "ood"  # + condition suffix
STREAM_SUPPORT_POOL = "support_pool"
STREAM_CONFIGURER = "configurer"
STREAM_MODEL_INIT = "model_init"
STREAM_TASK_ORDER = "task_order"
STREAM_REPLAY = "replay"
STREAM_RANDOM_MASK = "random_mask"


def _key_ints(*keys) -> list[int]:
    out = []
    for k in keys:
        if isinstance(k, str):
            out.append(zlib.crc32(k.encode()))
        elif isinstance(k, (int, np.integer)):
            out.append(int(k) & 0xFFFFFFFF)
        else:
            raise TypeError(f"bad seed key {k!r}")
    return out


def seed_seq(*keys) -> np.random.SeedSequence:
    return np.random.SeedSequence(_key_ints(*keys))


def rng(*keys) -> np.random.Generator:
    return np.random.default_rng(seed_seq(*keys))


def seed_id(*keys) -> int:
    """Stable 64-bit identity for an instance (used for support/query disjointness)."""
    st = seed_seq(*keys).generate_state(2, dtype=np.uint32)
    return int(st[0]) << 32 | int(st[1])


def torch_seed(*keys) -> int:
    return seed_id(*keys) & 0x7FFFFFFFFFFFFFFF
