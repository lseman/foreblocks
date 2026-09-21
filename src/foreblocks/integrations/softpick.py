"""Optional external Softpick attention backend.

Backend loading stays here so attention modules do not depend on a vendored
implementation inside the transformer package. Callers retain their fallback
when the optional ``flash_softpick_attn`` module is unavailable.
"""

from importlib import import_module


def load_parallel_softpick_attention():
    module = import_module("flash_softpick_attn")
    return module.parallel_softpick_attn
