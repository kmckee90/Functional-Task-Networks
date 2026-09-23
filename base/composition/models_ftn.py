"""models_ftn.py - construct a base FTN backbone from a compose_config Config.

Same dispatch as base/run_compose.build_model, kept here so the composition
drivers don't import base run_compose (which would collide with a moved
composition/run_compose.py). Imports the model classes from base/ via _paths.
"""

import _paths  # noqa: F401
from ftn import FTN


def build_ftn(Config, input_size, output_size):
    arch = getattr(Config, "ARCHITECTURE", "base").lower()
    common = dict(input_size=input_size, output_size=output_size,
                  hidden_size=Config.HIDDEN_SIZE)
    if arch == "lora":
        from lora_ftn import LoRAFTN
        return LoRAFTN(**common, **Config.LORA_PARS)
    if arch in ("stack", "stack_residual", "rnn", "rnn_residual"):
        from stack_ftn import StackFTN
        return StackFTN(
            **common,
            recurrent=arch in ("rnn", "rnn_residual"),
            residual=arch in ("stack_residual", "rnn_residual"),
            **Config.STACK_PARS,
        )
    if arch == "lorarnn":
        from lora_rnn_ftn import LoRARNNFTN
        return LoRARNNFTN(**common, **Config.LORARNN_PARS)
    if arch == "hml":
        from hml_ftn_stable import HMLFTN
        return HMLFTN(**common, **Config.HML_PARS)
    if arch == "quad":
        return FTN(**common, **Config.QUAD_PARS)
    return FTN(**common, **Config.FTN_PARS)
