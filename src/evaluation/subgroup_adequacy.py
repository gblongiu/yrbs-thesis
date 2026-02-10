from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SubgroupAdequacy:
    n: int
    n_pos: int
    n_neg: int
    event_rate: float
    adequate: bool
    reason: str


def evaluate_subgroup_adequacy(
    y_true: np.ndarray,
    *,
    min_group_n: int,
    min_group_pos: int,
    min_group_neg: int,
    min_group_eventrate: Optional[float] = None,
) -> SubgroupAdequacy:
    y = np.asarray(y_true, dtype=int)
    n = int(y.size)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    event_rate = float(n_pos / n) if n > 0 else np.nan

    reasons = []
    if n < int(min_group_n):
        reasons.append(f"n<{int(min_group_n)}")
    if n_pos < int(min_group_pos):
        reasons.append(f"pos<{int(min_group_pos)}")
    if n_neg < int(min_group_neg):
        reasons.append(f"neg<{int(min_group_neg)}")
    if min_group_eventrate is not None and not np.isnan(event_rate):
        if event_rate < float(min_group_eventrate):
            reasons.append(f"event_rate<{float(min_group_eventrate):.4f}")

    reason = ";".join(reasons)
    return SubgroupAdequacy(
        n=n,
        n_pos=n_pos,
        n_neg=n_neg,
        event_rate=event_rate,
        adequate=(len(reasons) == 0),
        reason=reason,
    )
