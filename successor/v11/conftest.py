"""Test configuration.

`HLWM_TEST_ADAPTER_MODE={lora,dora,pissa}` re-runs the whole suite with that
adapter decomposition as the default, so the v10 test battery is a real
regression check on the dora/pissa paths rather than on `lora` three times.
"""

import os

from modeling_hlwm import HLWMConfig

_MODE = os.environ.get("HLWM_TEST_ADAPTER_MODE")
if _MODE:
    if _MODE not in ("lora", "dora", "pissa"):
        raise ValueError(f"HLWM_TEST_ADAPTER_MODE must be lora|dora|pissa, got {_MODE!r}")

    # The dataclass `__init__` captured the field default at class creation, so
    # rebinding the field object would not reach `cls(**values)`. Wrap the
    # constructor instead; an explicit caller-supplied mode still wins.
    _original_init = HLWMConfig.__init__

    def _init(self, *args, **kwargs):
        if not args:
            kwargs.setdefault("adapter_mode", _MODE)
        _original_init(self, *args, **kwargs)

    HLWMConfig.__init__ = _init
