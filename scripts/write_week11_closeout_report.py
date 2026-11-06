"""Write the human-readable closeout from sealed analysis and control evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

from eventground_wam.b0_training import read_sealed
from eventground_wam.week11_analysis import ARMS


def write_report(root):
    remote = root/'remote'
    analysis = read_sealed(remote/'analysis.json')
    metadata = read_sealed(root/'metadata-verification.json')
    secondary = read_sealed(remote/'secondary-statistics.json')
    controls = {p: read_sealed(remote/'controls'/(p+'.json')) for p in ('targeted-467', 'clean-200')}
    lines = [
        '# Week 11 formal closeout', '',
        '**Decision: no positive progression recommendation for adaptive v1. Keep M4-fixed as the current baseline.**', '',
        ('All 2,001 formal episodes completed on October 25, 2026, at 16:25 PDT, after approximately '
        '4 hours 34 minutes. The independent closeout revalidated every artifact and the frozen statistics. '
        'Targeted performance is lower with an entirely negative paired interval; clean performance is lower '
        'and its cluster interval reaches zero. Both original performance gates fail. Latency, resource and '
        'artifact/input gates pass. No training, new rollouts, mapping changes or automatic follow-up were performed.'), '',
        '## Outcomes and paired uncertainty', '',
        '| Variant | Targeted successes | Clean successes | Targeted p95 | Clean p95 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for arm in ARMS:
        target, clean = (analysis['summaries'][arm+'/'+p] for p in ('targeted-467', 'clean-200'))
        lines.append(f"| {arm} | {target['successes']}/467 ({target['success_rate']*100:.2f}%) | "
                     f"{clean['successes']}/200 ({clean['success_rate']*100:.2f}%) | "
                     f"{target['latency']['all_queries']['p95_ms']:.2f} ms | {clean['latency']['all_queries']['p95_ms']:.2f} ms |")
    lines += ['', '| Panel | Contrast (left − right) | Difference, pp | Paired 95% CI, pp | Left-only / right-only wins | Both succeed / both fail |',
              '| --- | --- | ---: | --- | ---: | ---: |']
    for panel, contrasts in analysis['statistics'].items():
        for contrast, row in contrasts.items():
            lo, hi = row['paired_95_ci_pp']
            lines.append(f"| {panel} | {contrast.replace(' minus ', ' − ')} | {row['difference_pp']:+.2f} | "
                f"[{lo:+.2f}, {hi:+.2f}] | {row['left_only_success']} / {row['right_only_success']} | "
                f"{row['both_success']} / {row['both_failure']} |")
    lines += ['', '![Paired effects and frozen confidence intervals](presentation-v2/figures/paired-effects.png)', '',
        ('Each panel uses 10,000 common-index PCG64 bootstrap draws across all three contrasts, with linear '
        '2.5/97.5 percentiles. Targeted resamples the 467 variant tasks with seed 7. Clean resamples the '
        '40 canonical tasks with seed 8, retaining all five states in every selected task. These are '
        'conditional descriptive panel intervals, not estimates of training-seed variability or unseen-task '
        'generalization. The exact zero control interval reflects identical observed outcomes; it is not '
        'a general equivalence guarantee. The clean interval includes zero.'), '',
        ('The adaptive policy rescues 19 targeted cases but loses 51, for a net loss of 32. On clean it '
        'rescues 7 and loses 17, for a net loss of 10. Comparison with the event-running fixed8 control '
        'produces the same paired results.'), '',
        '## Original decision gates', '',
        '| Criterion | Observed | Decision |', '| --- | --- | --- |',
        '| Targeted gain ≥ +2 pp | −6.85 pp | FAIL |',
        '| Targeted paired lower bound > 0 | −10.28 pp | FAIL |',
        '| Clean difference ≥ −2 pp | −5.00 pp | FAIL |',
        '| Clean paired lower bound ≥ −5 pp | −10.00 pp | FAIL |',
    ]
    for panel, v in analysis['latency_gates'].items():
        lines.append(f"| {panel}: p95 ≤ 200 ms and ≤ 2.5× fixed | {v['adaptive_p95_ms']:.2f} ms; "
                     f"{v['adaptive_to_fixed_p95_ratio']:.3f}× | PASS |")
    lines += ['| Complete artifacts and all required pairwise initial inputs | All 2,001 episodes validated; all comparisons exact | PASS |',
              '| Original time, output, disk and memory limits | All within limits | PASS |', '',
        ('The result is an observed performance regression under this frozen integration. The measured '
        'latency is acceptable under the stated engineering limits; it does not compensate for the '
        'performance failures. No prospective criterion was changed after seeing outcomes.'), '',
        '## Fixed-control episode and trajectory audit', '',
        '| Check | Targeted | Clean | Total |', '| --- | ---: | ---: | ---: |']
    a, b = (controls[p]['summary'] for p in ('targeted-467', 'clean-200'))
    metrics = [
        ('Episode pairs checked', 'episode_count'),
        ('Fully bitwise-equal non-timing trajectories', 'fully_bitwise_equal_trajectories'),
        ('Episodes with policy-query input differences', 'episodes_with_query_input_differences'),
        ('Episodes with action differences', 'episodes_with_action_differences'),
        ('Identical-input queries compared', 'equal_input_query_count'),
        ('Different actions on identical inputs', 'equal_input_queries_with_different_actions'),
    ]
    for label, key in metrics:
        lines.append(f'| {label} | {a[key]} | {b[key]} | {a[key]+b[key]} |')
    for label, key in [('Outcome disagreements', 'outcome_disagreements'), ('Episode-length disagreements', 'episode_length_disagreements')]:
        lines.append(f'| {label} | {len(a[key])} | {len(b[key])} | {len(a[key])+len(b[key])} |')
    pairs = [p for c in controls.values() for p in c['pairs']]
    render_only = [p for p in pairs if not p['all_non_timing_steps_bitwise_equal']
                   and p['first_input_difference_query'] is None
                   and {k for k, v in p['fields'].items() if not v['all_rows_bitwise_equal']} <= {'agent_rgb', 'wrist_rgb'}]
    contexts = [p['first_input_difference_context'] for p in pairs if p['first_input_difference_context']]
    max_delta = max(x['max_abs_delta'] for c in contexts for x in c['camera_differences'].values())
    max_channels = max(sum(x['different_channel_values'] for x in c['camera_differences'].values()) for c in contexts)
    lines += ['', ("All **667 success/failure pairs match**, and all **16,500 identical-input queries produce bitwise-identical decoded action chunks**. "
        'The query input comparison includes both recorded uint8 cameras, the float32 proprioception consumed '
        'by the action adapter, and verified identical instructions. Every non-timing dataset was compared; '
        'unequal-length tails were retained.'), '',
        (f"Of the 70 trajectories that are not entirely bitwise identical, {len(render_only)} differ only in camera "
        'frames outside policy-query observations. The other 24 have policy-query camera differences and action '
        'differences. At each first differing query, simulator state, raw/effective proprioception and all prior '
        f'executed actions still match. The first camera differences are at most **{max_delta} intensity level**, '
        f'affecting at most **{max_channels} channel values**. This is consistent with small rendering variation '
        'preceding action divergence; it does not identify the rendering mechanism or prove a causal explanation '
        'for the adaptive regression.'), '',
        ('Eight of these pairs have different successful episode lengths. Their total query counts differ by '
        '+5 on targeted and +4 on clean in event/fixed8. All eight preserve the same binary outcome. Thus the '
        'fixed8 control is outcome-equivalent on this panel, while full-trajectory equality must be qualified. '
        'There is no observed same-input action discrepancy from running the event branch.'), '',
        '| Panel / zero-based pair | Suite / runtime task | Fixed steps | Event/fixed8 steps | First differing query step | Changed camera channel values |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for panel, c in controls.items():
        for p in c['pairs']:
            if p['left_policy_steps'] == p['right_policy_steps']:
                continue
            ctx = p['first_input_difference_context']
            channels = sum(x['different_channel_values'] for x in ctx['camera_differences'].values())
            lines.append(f"| {panel} / {p['ordinal']} | {p['task']['suite']} / {p['task']['runtime_task_index']} | "
                         f"{p['left_policy_steps']} | {p['right_policy_steps']} | {ctx['recorded_step']} | {channels} |")
    lines += ['', ('Full field-by-field differences and first-divergence contexts are preserved in '
        '[targeted control audit](remote/controls/targeted-467.json) and [clean control audit](remote/controls/clean-200.json).'), '',
        '## Predefined descriptive subsets', '',
        ('These one-way targeted subsets use the prescribed seed-7, 10,000-draw bootstrap within each subset. '
        'There is no multiplicity adjustment and no subset replaces the primary endpoint. The complete '
        '[machine-readable table](remote/secondary-statistics.json) includes all three contrasts. Since the '
        'fixed controls match every binary outcome, both adaptive-versus-control contrasts match in every subset. '
        'The contact subset is the frozen task-description proxy, not validated contact-density labels.'), '',
        '| Group | Subset | n | Fixed successes | Adaptive successes | Difference, pp | Paired 95% CI, pp |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for group, subsets in secondary.items():
        if group == 'digest':
            continue
        for label, contrasts in subsets.items():
            row = contrasts['M4-EG-adaptive minus M4-fixed']; lo, hi = row['paired_95_ci_pp']
            lines.append(f"| {group} | {label} | {row['count']} | {row['right_successes']} | {row['left_successes']} | "
                         f"{row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    if 'Long/unlabeled' not in secondary['difficulty']:
        lines += ['| difficulty | Long/unlabeled | 0 | 0 | 0 | — | Not estimable |', '',
                  ('All 467 selected targeted tasks have numeric difficulty labels 1–5. The predefined '
                   'Long/unlabeled stratum is empty; no task was dropped and no empty-sample interval is estimated.')]
    lines += ['', '## Timing and horizon evidence', '',
        ('All first, subsequent and failed-query costs are retained; there were no query exceptions or retries. '
        'Per-arm reports retain pooled median/p95, first/subsequent timing, equal-episode-weight summaries, '
        'component timing, model load, simulator and recording costs. The event-running fixed8 control '
        'adds cost while preserving all binary outcomes.'), '',
        '| Adaptive panel | h=2 | h=4 | h=8 | h=16 | h=32 | Valid-none fallbacks to 8 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for panel in controls:
        q = read_sealed(remote/'panels/M4-EG-adaptive'/(panel+'.json'))['query_summary']
        lines.append('| '+panel+' | '+' | '.join(str(q['used_horizons'].get(str(h), 0)) for h in (2, 4, 8, 16, 32))
                     +f" | {q['fallbacks'].get('no_new_onset', 0)} |")
    lines += ['', ('These are deployment frequencies, not event-prediction accuracy or causal attribution. '
        'The previously frozen recorded-dev mapping diagnostic retains its weak-label scope; it was not '
        'used to change the mapping. On-policy label accuracy and independent label QA remain unverified.'), '',
        '## Preserved examples, resources and reproducibility', '',
        ('The [selected-case index](remote/selected-case-index.json) takes the first manifest ordinal in each '
        'paired outcome class on each panel. [Rendered examples](presentation-v2/presentation.json) show both cameras '
        'at the first policy query, midpoint and last recorded pre-action observation for all three arms. '
        'The selection does not favor visually persuasive examples and does not assert failure causes.'), '',
        '- [Targeted fixed-only success example](presentation-v2/figures/targeted-467-fixed_only_success.png)',
        '- [Targeted adaptive-only success example](presentation-v2/figures/targeted-467-adaptive_only_success.png)',
        '- [Clean fixed-only success example](presentation-v2/figures/clean-200-fixed_only_success.png)',
        '- [Clean adaptive-only success example](presentation-v2/figures/clean-200-adaptive_only_success.png)', '',
    ]
    resources = analysis['analysis_resources']
    lines += [(f"The accepted CPU analysis took {resources['analysis_active_seconds']:.2f} active seconds. The stopped "
        f"first attempt's {resources['prior_analysis_attempts'][0]['active_seconds']:.2f} seconds remain charged. "
        f"Qualification, evaluation and analysis total **{resources['total_active_seconds']/3600:.3f} active hours** "
        'before the separately recorded presentation-rendering cost, well within the 48-hour limit. All '
        'panel totals remain below 12 hours. The overall highest GPU reservation is the preserved pilot peak '
        'of 23.78 GiB (75.9%); the formal run peaks at 15.19 GiB. Analysis makes zero GPU inference calls. '
        f"Analysis process-tree RSS peaks at {resources['analysis_peak_process_tree_rss_bytes']/1024**3:.2f} GiB; "
        f"free disk at close is {resources['free_bytes']/1024**3:.2f} GiB, above the 50-GiB reserve."), '',
        (f"All **{len(read_sealed(remote/'raw-inventory.json')['files']):,} raw evaluation files**, including **2,001 HDF5 recordings**, "
        'remain unchanged on Feifei under the complete [SHA-256 inventory](remote/raw-inventory.json). '
        f"All **{metadata['metadata_files_verified']:,} metadata files** ({metadata['metadata_uncompressed_bytes']:,} bytes) "
        'are mirrored in [formal-metadata.tar.gz](formal-metadata.tar.gz), and every decompressed file matches '
        'the remote inventory. The 76-GB raw recording directory was not downloaded or deleted. The local '
        'metadata-derived statistics exactly reproduce the remote analysis.'), '',
        ('The accepted analysis source is [package-v3](package-v3/package.json), extending the executed formal '
        'archive without replacing an inherited byte. Its 27 focused tests pass both locally and on the pinned '
        'Feifei Python/NumPy runtime. Package v1 stopped on nested timing aggregation; package v2 remained local '
        'after a test-import issue; both sources and test/attempt records are retained. These issues affected '
        'postprocessing only. The original 79-test launch qualification and all prior freezes remain unchanged.'), '',
        ('The [analysis method and rerun command](../../WEEK11_CLOSEOUT_METHOD.md), [paired matrices](remote/paired-matrices.json), '
        '[sealed analysis](remote/analysis.json), source archives, metadata archive, figures, tests, and '
        'before/after status snapshots form the closeout package. This closes the requested formal analysis '
        'and fixed-control audit. Adaptive failure-mechanism diagnosis and any new experiment remain separate work.'), '',
        f"Analysis digest: `{analysis['digest']}`.  ",
        f"Accepted analysis source digest: `{analysis['analysis_source_digest']}`.  ",
        f"Formal source digest: `{analysis['source_digest']}`.  ",
        f"Raw inventory digest: `{analysis['raw_inventory_digest']}`.", '',
    ]
    (root/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    write_report(parser.parse_args().root)
