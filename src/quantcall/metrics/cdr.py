from __future__ import annotations


def compute_cdr(free_fcr: float, constrained_fcr: float, baseline_fcr: float) -> float:
    """Constrained Decoding Recovery: fraction of baseline gap recovered.

    CDR = (constrained_fcr - free_fcr) / (baseline_fcr - free_fcr)

    Clamped to [0, 1]. When free_fcr == baseline_fcr (no degradation), returns 1.0
    if constrained >= baseline, else 0.0.
    """
    gap = baseline_fcr - free_fcr
    if abs(gap) < 1e-12:
        return 1.0 if constrained_fcr >= baseline_fcr else 0.0
    raw = (constrained_fcr - free_fcr) / gap
    return max(0.0, min(1.0, raw))
