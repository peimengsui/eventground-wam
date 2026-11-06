"""Week 9 common-resample estimates; no selection, missing-arm imputation, or p-values."""
from __future__ import annotations

import numpy as np

ARMS = ('B0', 'M1', 'M2', 'M3', 'M4')
CONTRASTS = {
    'M1_minus_B0': (-1, 1, 0, 0, 0),
    'M2_minus_B0': (-1, 0, 1, 0, 0),
    'M3_minus_B0': (-1, 0, 0, 1, 0),
    'M4_minus_B0': (-1, 0, 0, 0, 1),
    'M2_minus_M1': (0, -1, 1, 0, 0),
    'M3_minus_M1': (0, -1, 0, 1, 0),
    'M4_minus_M2': (0, 0, -1, 0, 1),
    'M4_minus_M3': (0, 0, 0, -1, 1),
    'latent_minus_pixel_main_effect': (0, -.5, .5, -.5, .5),
    'coupling_main_effect': (0, -.5, -.5, .5, .5),
    'interaction': (0, 1, -1, -1, 1),
}


def _matrix(outcomes):
    if set(outcomes) != set(ARMS):
        raise ValueError('complete B0/M1/M2/M3/M4 matrix required')
    arrays = [np.asarray(outcomes[arm], dtype=float) for arm in ARMS]
    if (any(a.shape != arrays[0].shape or not np.isin(a, (0, 1)).all() for a in arrays)
            or arrays[0].size == 0):
        raise ValueError('aligned, complete binary outcomes required; failures remain zero')
    return np.stack(arrays, axis=-1)


def bootstrap_rates(outcomes, *, panel, draws=10000):
    """Perturbed task resampling; clean canonical-task clusters retain all 50 states."""
    values = _matrix(outcomes)
    if panel == 'perturbed':
        if values.ndim != 2:
            raise ValueError('perturbed outcomes must be task vectors')
        seed = 7
    elif panel == 'clean':
        if values.ndim != 3 or values.shape[1] != 50:
            raise ValueError('clean outcomes must have canonical task x 50 state shape')
        values = values.mean(axis=1)
        seed = 8
    else:
        raise ValueError('unknown panel')
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(values), (draws, len(values)))
    # Draw exactly one index matrix, shared by all policies and every contrast.
    sampled = np.stack([values[:, i][indices].mean(axis=1) for i in range(len(ARMS))], axis=1)
    return values.mean(axis=0), sampled


def _estimate(point, samples, scale=1.):
    return {'estimate': float(point * scale),
            'percentile_95': (np.quantile(samples, [.025, .975], method='linear') * scale).tolist()}


def summarize_panel(outcomes, *, panel, draws=10000):
    means, samples = bootstrap_rates(outcomes, panel=panel, draws=draws)
    return {
        'success_rates': {arm: _estimate(means[i], samples[:, i]) for i, arm in enumerate(ARMS)},
        'contrasts_percentage_points': {
            name: _estimate(means @ weights, samples @ weights, 100.)
            for name, weights in CONTRASTS.items()},
        'bootstrap': {'draws': draws, 'rng': 'PCG64', 'seed': 7 if panel == 'perturbed' else 8,
                      'unit': 'variant_task' if panel == 'perturbed' else 'canonical_task_50_states',
                      'common_indices': True, 'quantile_method': 'linear'},
    }


def robustness_ratios(perturbed, clean, *, draws=10000):
    numerator, num_draws = bootstrap_rates(perturbed, panel='perturbed', draws=draws)
    denominator, den_draws = bootstrap_rates(clean, panel='clean', draws=draws)
    result = {}
    for i, arm in enumerate(ARMS):
        invalid = int((den_draws[:, i] == 0).sum())
        result[arm] = {
            'estimate': float(numerator[i] / denominator[i]) if denominator[i] else None,
            'percentile_95': np.quantile(num_draws[:, i] / den_draws[:, i], [.025, .975],
                method='linear').tolist() if denominator[i] and not invalid else None,
            'zero_denominator_draws': invalid,
            'interpretation': 'aggregate panel ratio; distinct task distributions, not causal robustness',
        }
    return result
