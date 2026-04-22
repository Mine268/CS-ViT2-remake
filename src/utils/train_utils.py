"""
Training utility functions.
"""


def get_progressive_dropout(
    step: int,
    total_steps: int,
    warmup_steps: int = 10000,
    target_dropout: float = 0.1,
) -> float:
    del total_steps
    if step < warmup_steps:
        return 0.0
    return float(target_dropout)
