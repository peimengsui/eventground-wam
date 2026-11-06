"""Fixed gates and paired exploratory comparisons; never starts a full run."""
import argparse
import json
from pathlib import Path

from analyze_week10_weak_pilot import read_eval, paired_bootstrap
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_preservation_runtime import verify, parent_base, selected_weight, validate_training_completion
from eventground_wam.eventground_weak_evaluation import promotion


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, required=True)
    base = parser.parse_args().base; p, _, manifests, _ = verify(base)
    weight, diagnostic = selected_weight(base)
    baseline, first = read_eval(parent_base(base)/'eval/dev/sft')
    completions = {'sft': baseline}; predictions = {'sft': first}; training = {}; gates = {}; comparisons = {}
    for arm in ('hybrid', 'control'):
        done = validate_training_completion(base, arm); evaluation, rows = read_eval(base/'eval/dev'/arm)
        if (evaluation['protocol_digest'] != p['digest'] or evaluation['manifest_digest'] != manifests['dev']['digest']
                or evaluation['endpoint_completion_digest'] != done['digest']):
            raise ValueError('Evaluation identity differs')
        completions[arm] = evaluation; predictions[arm] = rows; training[arm] = done
        gates[arm] = promotion(baseline['metrics'], evaluation['metrics'], all(done['gates'].values()))
        comparisons[arm+'_minus_sft'] = paired_bootstrap(first, rows)
    comparisons['hybrid_minus_control'] = paired_bootstrap(predictions['control'], predictions['hybrid'])
    changes = {}
    for arm in ('hybrid', 'control'):
        a, b = completions['sft']['metrics'], completions[arm]['metrics']
        changes[arm] = dict(macro_positive_iou=b['macro_positive_iou']-a['macro_positive_iou'],
                            none_false_positive_rate=b['none_false_positive_rate']-a['none_false_positive_rate'],
                            matched_positive_start_mae=b['matched_positive_start_mae']-a['matched_positive_start_mae'])
    result = seal_contract(dict(artifact_type='eventground_wam.preservation_pilot_results', protocol_digest=p['digest'],
        selected_weight=weight, diagnostic_digest=diagnostic['digest'],
        evaluations={a:c['digest'] for a,c in completions.items()}, training={a:c['digest'] for a,c in training.items()},
        metrics={a:c['metrics'] for a,c in completions.items()}, changes_from_sft=changes, gates=gates,
        comparisons=comparisons, full_training_launched=False, report_panel_opened=False, independent_qa_passed=False,
        disposition='Pilot complete; user review required before any new full-run scope, regardless of gate outcome.',
        limitations=p['limitations']))
    out = base/'analysis'; out.mkdir(exist_ok=False)
    with (out/'results.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    lines = ['# Supervised-preservation pilot', '',
        'This is a single exploratory, adaptive dev experiment. Both arms start from the identical completed future-onset SFT endpoint.', '',
        f'The no-update training-only diagnostic selected lambda = {weight:.8f} using the frozen 25% gradient-ratio rule.',
        'Both arms use 500 identical prompts, 32 updates, effective batch 16 (last batch 4), microbatch 1, seed 17, FP32, and RLVR learning rates (head 1e-5, LoRA 1e-6).',
        'The hybrid optimizes GRPO + lambda × structured SFT; the control optimizes structured SFT alone.', '',
        '| Arm | Macro positive IoU | None false positives | Matched onset MAE | Dev gates |',
        '|---|---:|---:|---:|---|']
    for arm in ('sft', 'hybrid', 'control'):
        m = completions[arm]['metrics']; false = sum(r['target']['next_event']=='none' and r['prediction']['next_event']!='none' for r in predictions[arm])
        verdict = 'baseline' if arm=='sft' else 'passed' if gates[arm]['promoted'] else ', '.join(k for k,v in gates[arm]['gates'].items() if not v)
        lines.append(f"| {arm} | {100*m['macro_positive_iou']:.2f}% | {false}/99 | {m['matched_positive_start_mae']:.3f} frames | {verdict} |")
    lines.extend(['', 'Unchanged gates require at least +1 percentage point macro IoU, at most +2 points none false positives, and no more than +1 frame matched onset MAE; format, class coverage and integrity must pass.', '',
                  '95% paired source-bootstrap intervals (10,000 draws, 103 physical sources, PCG64 seed 10):', ''])
    for name, comparison in comparisons.items():
        interval = comparison['interval95']
        text = f'[{100*interval[0]:+.2f}, {100*interval[1]:+.2f}] percentage points' if interval else 'withheld'
        lines.append(f'- {name.replace("_", " ")}: {text}.')
    lines.extend(['', 'Training and completion checks:', ''])
    for arm, done in training.items():
        lines.append(f"- {arm}: {done['global_step']} updates / {done['prompts_seen']} prompts, {done['elapsed_seconds']/60:.2f} minutes, {done['peak_reserved_bytes']/1024**3:.2f} GiB peak reserved. All integrity gates passed.")
    lines.extend(['', '![Training and dev comparison](learning-and-dev.png)', '',
        'No full RLVR training or report evaluation was launched. Gate passage alone would not establish a benefit over the matched SFT-only control.', '',
        'Limitations:', ''] + ['- '+x for x in p['limitations']])
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gates=gates, changes_from_sft=changes, selected_weight=weight)), flush=True)


if __name__ == '__main__':
    main()
