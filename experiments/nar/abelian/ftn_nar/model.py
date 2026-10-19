"""Shim: the canonical copies live in the shared nar/ftn_core package."""

from ftn_core.model import (  # noqa: F401
    FTN2,
    MaskedStage,
    NeuronBank,
    accuracy,
    sequence_ce,
)
