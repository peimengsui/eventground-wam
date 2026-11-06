import numpy as np
import pytest

from eventground_wam.factorial_analysis import ARMS, bootstrap_rates, robustness_ratios, summarize_panel


def test_known_factorial_effects_and_common_indices():
    outcomes = dict(zip(ARMS, ([0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0],
                              [1, 1, 1, 0], [1, 1, 1, 1])))
    summary = summarize_panel(outcomes, panel='perturbed')
    effects = summary['contrasts_percentage_points']
    assert effects['coupling_main_effect']['estimate'] == 50
    assert effects['latent_minus_pixel_main_effect']['estimate'] == 25
    assert effects['interaction']['estimate'] == 0
    indices = np.random.Generator(np.random.PCG64(7)).integers(0, 4, (10000, 4))
    delta = np.array(outcomes['M4'])-np.array(outcomes['M2'])
    assert effects['M4_minus_M2']['percentile_95'] == np.quantile(
        delta[indices].mean(1)*100, [.025, .975], method='linear').tolist()


def test_clean_clusters_keep_fifty_states_together():
    values = np.stack([np.zeros(50), np.ones(50)])
    outcomes = {arm: values for arm in ARMS}
    means, draws = bootstrap_rates(outcomes, panel='clean')
    assert np.all(means == .5)
    assert set(np.unique(draws)) == {0, .5, 1}
    assert np.array_equal(draws[:, 0], draws[:, 4])
    assert summarize_panel(outcomes, panel='clean')['contrasts_percentage_points']['interaction'] == {
        'estimate': 0., 'percentile_95': [0., 0.]}


def test_zero_ratio_denominator_is_explicit_never_silently_drop_draws():
    perturbed = {arm: [1, 0] for arm in ARMS}
    clean = {arm: np.zeros((2, 50)) for arm in ARMS}
    result = robustness_ratios(perturbed, clean)
    assert result['M4']['estimate'] is None
    assert result['M4']['percentile_95'] is None
    assert result['M4']['zero_denominator_draws'] == 10000


@pytest.mark.parametrize('bad', [{'M1': [1]}, {arm: [1, np.nan] for arm in ARMS}])
def test_rejects_incomplete_matrix(bad):
    with pytest.raises(ValueError):
        summarize_panel(bad, panel='perturbed')
