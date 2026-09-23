# Regression guard for the exposure-ratio truncation (exposure_ratio_clip):
# the estimated ratio e_bar_hat^pi / e_hat in the LE-IIPS / ED-DR weight
# denominator is capped at M by flooring the denominator (truncated-IPS
# style). At small n the EM fit routinely collapses e_hat toward the
# numeric floor on a sizable share of records (measured ~10% of records on
# average, max 17%, at n=100 — the norm there, not a rare guard), which
# otherwise inflates single weights by up to 1 / PROB_CLIP and dominates
# the estimate.
import numpy as np

from estimators_slate import clip_exposure_denominator


def test_clip_disabled_returns_denominator_unchanged():
    e_hat = np.array([0.5, 1e-8, 0.9])
    e_bar = np.array([0.4, 0.6, 0.9])
    out = clip_exposure_denominator(
        exposure_factual_hat=e_hat,
        expected_exposure_eval_hat=e_bar,
        exposure_ratio_clip=None,
    )
    assert out is e_hat


def test_clip_caps_ratio_at_m_and_leaves_small_ratios_alone():
    M = 10.0
    e_hat = np.array([0.5, 1e-6, 0.05])  # ratios: 0.8, 6e5, 10 (boundary)
    e_bar = np.array([0.4, 0.6, 0.5])
    out = clip_exposure_denominator(
        exposure_factual_hat=e_hat,
        expected_exposure_eval_hat=e_bar,
        exposure_ratio_clip=M,
    )
    ratio = e_bar / out
    assert np.all(ratio <= M + 1e-12)
    # non-binding entries keep the raw denominator (ratio unchanged)
    assert out[0] == e_hat[0]
    assert np.isclose(ratio[2], M)
    # binding entry is floored exactly at e_bar / M
    assert np.isclose(out[1], e_bar[1] / M)
