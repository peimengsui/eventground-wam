"""Prospective engineering decision for the selected 27-case continuation screen."""
from __future__ import annotations

MODELS = ('original', 'h8-continuation', 'mixed-continuation')
CONDITIONS = tuple(f'{m}/{r}' for m in MODELS for r in ('fixed8', 'adaptive'))


def screen_decision(cases, outcomes, *, integrity_valid):
    ids = [c['id'] for c in cases]
    if (len(ids) != 27 or len(set(ids)) != 27 or set(outcomes) != set(CONDITIONS)
            or type(integrity_valid) is not bool):
        raise ValueError('screen must contain the exact 27-case six-condition matrix')
    for rows in outcomes.values():
        if set(rows) != set(ids) or any(type(v) is not bool for v in rows.values()):
            raise ValueError('screen has missing, extra or non-binary outcomes')
    counts = {c: sum(rows.values()) for c, rows in outcomes.items()}
    gaps = {m: counts[m+'/adaptive']-counts[m+'/fixed8'] for m in MODELS}
    checks = {'all_input_integrity_and_resource_checks_valid': integrity_valid,
              'adaptive_gain_over_h8_at_least_two': counts['mixed-continuation/adaptive']-counts['h8-continuation/adaptive'] >= 2,
              'adaptive_gain_over_original_at_least_two': counts['mixed-continuation/adaptive']-counts['original/adaptive'] >= 2,
              'fixed8_loss_vs_h8_at_most_one': counts['mixed-continuation/fixed8']-counts['h8-continuation/fixed8'] >= -1,
              'fixed8_loss_vs_original_at_most_one': counts['mixed-continuation/fixed8']-counts['original/fixed8'] >= -1,
              'adaptive_gap_improves_vs_h8_at_least_one': gaps['mixed-continuation']-gaps['h8-continuation'] >= 1}
    contrasts = {}
    for left, right in (
        ('mixed-continuation/adaptive','h8-continuation/adaptive'),
        ('mixed-continuation/adaptive','original/adaptive'),
        ('mixed-continuation/fixed8','h8-continuation/fixed8'),
        ('mixed-continuation/fixed8','original/fixed8'),
        *((m+'/adaptive',m+'/fixed8') for m in MODELS),
    ):
        wins=[i for i in ids if outcomes[left][i] and not outcomes[right][i]]
        losses=[i for i in ids if outcomes[right][i] and not outcomes[left][i]]
        contrasts[left+' minus '+right]={'net_successes':len(wins)-len(losses),'wins':wins,'losses':losses}
    return {'counts':counts,'adaptive_minus_fixed8_counts':gaps,'contrasts':contrasts,
            'interaction_gain':gaps['mixed-continuation']-gaps['h8-continuation'],
            'checks':checks,'eligible_to_propose_untouched_panel':all(checks.values()),
            'population_benefit_established':False,'baseline':'original M4-fixed'}
