"""Adapter stub for the original Powderworld implementation.

The Stage 0 code is written against a small environment interface:

- ``sample_state() -> grid/state``
- ``make_action_bank(k) -> List[Action]``
- ``rollout(state, action, horizon) -> RolloutResult``
- renderers for observation channels

The current starter repo uses ``ToyPowderWorld`` so that the protocol can be
run immediately.  Replace this adapter with calls into the original
``kvfrans/powderworld`` code when you want to reproduce the diagnostic protocol
in the real environment.
"""

from __future__ import annotations


class PowderworldAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PowderworldAdapter is a placeholder. Use ToyPowderWorld for Stage 0, "
            "or implement this adapter against the original Powderworld repo."
        )
