from __future__ import annotations

from quantcall.metrics.cdr import compute_cdr


def test_cdr_full_recovery():
    cdr = compute_cdr(free_fcr=0.6, constrained_fcr=0.9, baseline_fcr=0.9)
    assert abs(cdr - 1.0) < 1e-9


def test_cdr_no_recovery():
    cdr = compute_cdr(free_fcr=0.6, constrained_fcr=0.6, baseline_fcr=0.9)
    assert abs(cdr - 0.0) < 1e-9


def test_cdr_partial_recovery():
    cdr = compute_cdr(free_fcr=0.6, constrained_fcr=0.75, baseline_fcr=0.9)
    assert abs(cdr - 0.5) < 1e-9


def test_cdr_zero_gap():
    """If no degradation from baseline, CDR = 1.0."""
    cdr = compute_cdr(free_fcr=0.9, constrained_fcr=0.9, baseline_fcr=0.9)
    assert abs(cdr - 1.0) < 1e-9


def test_cdr_overcorrection_clamped():
    """CDR cannot exceed 1.0 even if constrained > baseline."""
    cdr = compute_cdr(free_fcr=0.7, constrained_fcr=1.0, baseline_fcr=0.9)
    assert cdr <= 1.0


def test_cdr_negative_clamped():
    """CDR is clamped at 0.0 if constrained is worse than free."""
    cdr = compute_cdr(free_fcr=0.8, constrained_fcr=0.7, baseline_fcr=0.9)
    assert cdr >= 0.0


def test_cdr_baseline_equals_free():
    """If free_fcr == baseline_fcr no degradation to recover."""
    cdr = compute_cdr(free_fcr=0.9, constrained_fcr=0.95, baseline_fcr=0.9)
    assert cdr >= 0.0
