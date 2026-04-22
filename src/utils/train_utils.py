"""
Training utility functions.
"""


def get_progressive_dropout(
    step: int,
    total_steps: int,
    warmup_steps: int = 10000,
    target_dropout: float = 0.1,
) -> float:
    """
    Return the decoder dropout rate scheduled for the current optimization step.

    The current remake keeps this schedule intentionally simple: dropout is disabled during
    warmup and then jumps to the configured target value. The `total_steps` argument is kept
    in the signature so more complex schedules can be dropped in without changing callers.
    """
    del total_steps
    if step < warmup_steps:
        return 0.0
    return float(target_dropout)
