"""Reconstruct experiment history and distinguish measurements from hypotheses."""
import argparse
import json
from collections import Counter
from pathlib import Path

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def reward_diagnostics(base,arm,manifest):
    rows={r['prompt_id']:r for r in manifest['rows']}
    path=base/'runs'/arm/'rewards.json'
    groups=[r for r in json.loads(path.read_text())['groups'] if not r['recovery_replay']]
    result={}
    for kind in ('contact','grasp','gripper_close','none'):
        selected=[r for r in groups if rows[r['prompt_id']]['target']['next_event']==kind]
        equal=[r for r in selected if max(r['rewards'])==min(r['rewards'])]
        result[kind]=dict(groups=len(selected),equal_reward_groups=len(equal),
            all_correct_type_ties=sum(all(v['next_event']==kind for v in r['predictions']) for r in equal),
            all_wrong_type_ties=sum(all(v['next_event']!=kind for v in r['predictions']) for r in equal))
    return dict(rewards_sha256=sha(path),by_class=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    base=parser.parse_args().base;root=base.parent
    old=root/'week10-weak-rlvr-v1';future=root/'week10-future-onset-v1';pilot=root/'week10-preservation-pilot-v1'
    report=sealed(base/'analysis/results.json');future_p=sealed(future/'protocol.json');pilot_p=sealed(pilot/'protocol.json')
    original_sft=sealed(old/'evaluations/SFT/completion.json');original_rl=sealed(old/'evaluations/primary/completion.json')
    original_decision=sealed(old/'analysis/primary-decision.json')
    history=[dict(task='original conservative task',arm='SFT',dev_n=48,metrics=original_sft['metrics'],gate='baseline'),
             dict(task='original conservative task',arm='RLVR primary',dev_n=48,metrics=original_rl['metrics'],gate='passed')]
    for arm in ('sft','primary','calibration'):
        c=sealed(future/'eval/dev'/arm/'completion.json')
        history.append(dict(task='next-new-onset task',arm=arm,dev_n=396,metrics=c['metrics'],
                            gate='baseline' if arm=='sft' else 'failed: IoU gain and none FP'))
    for arm in ('hybrid','control'):
        c=sealed(pilot/'eval/dev'/arm/'completion.json')
        history.append(dict(task='next-new-onset task',arm=arm,dev_n=396,metrics=c['metrics'],
                            gate='passed' if arm=='control' else 'failed: IoU gain and none FP'))
    train=sealed(future/'train-manifest.json')
    diagnostics={arm:reward_diagnostics(future,arm,train) for arm in ('primary','calibration')}
    diagnostics['hybrid']=reward_diagnostics(pilot,'hybrid',train)
    primary=read_rows(future/'eval/dev/primary/predictions.jsonl');calibration=read_rows(future/'eval/dev/calibration/predictions.jsonl')
    if [r['prompt_id'] for r in primary]!=[r['prompt_id'] for r in calibration]:raise ValueError('Unpaired dev predictions')
    changes=sum(a['prediction']!=b['prediction'] for a,b in zip(primary,calibration,strict=True))
    hypotheses=[
        dict(hypothesis='SFT supplies more direct and consistent supervision than the current four-sample relative-reward objective.',
             evidence='SFT-only continuation passed dev. On primary/calibration, 117/125 none groups tie with all four candidates correct; hybrid 118/125. Ties have zero policy advantage; KL remains. Positive all-wrong groups also tie.',
             qualification='This is a plausible mechanism, not a causal attribution. Correct-none ties themselves are not errors, and cross-prompt updates plus other factors could cause drift.',
             next_test='Training-only audit of reward variance and candidate coverage before considering any revised sampling or anchoring pilot.'),
        dict(hypothesis='RLVR shifts event-type boundaries toward grasp while sacrificing contact and some none decisions.',
             evidence='Starting revised SFT identifies 54 contact / 79 grasp / 52 close; primary 50/88/55; calibration 49/88/55; hybrid 49/87/54. None false positives increase 2→4/99. The SFT-only control is 50/83/52 with 2/99 false positives.',
             qualification='These are dev associations. The hybrid-control source interval includes zero, and neither seed variability nor independent event truth is established.',
             next_test='Audit the changed none/contact cases and reward-component gradients on training data; stronger preservation could reduce drift but may converge toward SFT-only behavior.'),
        dict(hypothesis='The small reward reweighting was too weak to materially change the learned behavior.',
             evidence=f'Primary and calibration differ on only {changes}/396 full dev predictions; their none tie counts are identical. Weights changed from [.4,.25,.2,.15] to [.3,.35,.2,.15].',
             qualification='Only this one predefined reweighting was tested; it does not rule out a substantially different reward design.',
             next_test='Do not repeat a blind weight sweep. Inspect whether a proposed reward actually separates useful candidate behaviors before any run.'),
        dict(hypothesis='A larger supervised coefficient might improve retention but need not produce an RLVR advantage.',
             evidence='One training-gradient-selected coefficient, 0.27453765, raised dev IoU slightly over plain RLVR but retained 4/99 none false positives. SFT-only scored higher on dev.',
             qualification='No coefficient sweep or proof of an optimal weight. The report evaluation tests only the dev-selected SFT-only checkpoint.',
             next_test='Treat stronger preservation as a new bounded hypothesis, with a matched SFT-only control and a separately frozen scope.'),
        dict(hypothesis='Weak-label uncertainty and limited single-observation information may cap recoverable timing accuracy.',
             evidence='Both objectives use the same automatic targets. The revised task forecasts a new onset up to 32 frames ahead from current RGB/proprioception/instruction; independent missed-event recall is unmeasured.',
             qualification='Neither label noise nor missing temporal information has been isolated as the cause of these results.',
             next_test='Prioritize targeted independent label/error review; temporal context would be a separate input/experiment change, not a patch to the current comparison.')]
    unlikely=[
        'Longer training alone is not an evidence-backed repair: it might help, but might also amplify the observed type/none tradeoff; no full RLVR run was tested.',
        'A constant valid-format reward cannot discriminate candidates when every constrained output is valid; that constant cancels in group-centered advantages.',
        'Uniform reward rescaling mostly cancels in group-standardized advantages (apart from the fixed epsilon). Relative reward redesign is different.',
        'More recordings under the original no-ongoing-event rule did not resolve report grasp coverage: 80 additional verified sources produced only 5 eligible grasp prompts. Revising the task semantics restored coverage and required fresh SFT.',
        'A decreasing training loss or near-zero mean GRPO policy-loss scalar alone does not establish held-out improvement. Centered advantages can give a near-zero on-policy scalar while its gradient is nonzero.',
        'Passing weak-label dev/report metrics does not establish true event detection or robot-policy improvement.'
    ]
    value=seal_contract(dict(artifact_type='eventground_wam.rlvr_sft_retrospective',report_results_digest=report['digest'],
        original_decision_digest=original_decision['digest'],history=history,reward_diagnostics=diagnostics,
        primary_calibration_changed_predictions=changes,hypotheses=hypotheses,unsupported_shortcuts=unlikely,
        correction='The earliest 48-prompt RLVR pilot passed. The later 396-prompt revised-task RLVR variants failed. The earlier statement that none passed was too broad.',
        cross_task_comparison='Original and next-new-onset tasks/manifests differ; do not compare absolute scores across those rows.',
        causal_claim='No RLVR causal superiority or general ineffectiveness established; small pilots, one seed, weak labels and adaptive dev reuse.'))
    with (base/'analysis/retrospective.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    lines=['# RLVR versus SFT: evidence and hypotheses','',
        '**Correction:** the earliest RLVR pilot passed its small 48-prompt dev gate. The later RLVR variants on the revised 396-prompt task failed. Saying that none had passed was too broad.', '',
        'The first task excluded anchors where an event was already ongoing; the revised task predicts the earliest new onset even during another ongoing event. Physical labels and ambiguity exclusions were retained. Absolute scores across these task versions are not directly comparable.', '',
        '| Task / dev size | Experiment | Macro positive IoU | None FP | Outcome |', '|---|---|---:|---:|---|']
    for row in history:
        m=row['metrics'];none_n=m['per_class']['none']['n'];fp=round(m['none_false_positive_rate']*none_n)
        lines.append(f"| {row['task']} / {row['dev_n']} | {row['arm']} | {100*m['macro_positive_iou']:.2f}% | {fp}/{none_n} | {row['gate']} |")
    lines+=['',
        'Original SFT: fresh M4 + event head, two passes over 1,332 prompts, 168 updates. The original RLVR pilot used 404 prompts / 26 updates. Its +1.20-point dev gain had a source-bootstrap interval spanning zero; the original report had zero selected grasp examples, so full training and report inference did not proceed.',
        'Task revision: after the bounded 80-source coverage probe, allow forecasts during ongoing events. Fresh M4 SFT used 1,609 prompts / two passes / 202 updates. Revised primary and calibration RLVR each restarted independently from that completed SFT, 500 prompts / 32 updates, with four candidates/prompt.',
        'Preservation experiment: one hybrid pilot and one SFT-only control each restarted from the same completed revised SFT and used identical 500 prompts/order, 32 updates, effective batch 16/micro 1, seed 17, FP32 and learning rates (head 1e-5, LoRA 1e-6). Hybrid loss was GRPO+0.27453765×SFT, with the coefficient frozen by a no-update training-only gradient diagnostic. No pilot weights were reused for another training arm.', '',
        'The earliest original pilot had no executed matched SFT continuation control. The later hybrid/control comparison matches data, initialization, optimizer rates and updates, but not candidate-sampling compute or wall time. Full/main RLVR has never been launched.', '',
        'Current report comparison (no new training):','',
        '| Fixed 384-prompt report | Starting SFT | SFT-only continuation |','|---|---:|---:|',
        f"| Macro positive IoU | {100*report['metrics']['sft']['macro_positive_iou']:.2f}% | {100*report['metrics']['control']['macro_positive_iou']:.2f}% |",
        f"| None false positives | {round(96*report['metrics']['sft']['none_false_positive_rate'])}/96 | {round(96*report['metrics']['control']['none_false_positive_rate'])}/96 |", '',
        f"Report IoU change: {100*report['delta_iou']:+.2f} points. 95% source-bootstrap interval: [{100*report['bootstrap']['interval95'][0]:+.2f}, {100*report['bootstrap']['interval95'][1]:+.2f}] points.",
        'The report recordings are separate from event train/dev, but labels/counts and the starting-SFT baseline were previously inspected. This is exploratory weak-label agreement, not untouched confirmation.', '',
        'Working hypotheses (not proven mechanisms):','']
    for item in hypotheses:
        lines += [f"- **{item['hypothesis']}** Evidence: {item['evidence']} Limit: {item['qualification']} Next diagnostic: {item['next_test']}"]
    lines+=['','Changes unlikely to solve the problem without additional evidence:','']+['- '+x for x in unlikely]
    lines+=['','All numerical/identity checks and previous negative results remain preserved. Candidate diversity, a stronger supervised coefficient, different KL strength, richer temporal inputs, independent label corrections and multiple training seeds remain untested alternatives; none is authorized or automatically launched by this retrospective.']
    (base/'analysis/RLVR_SFT_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(history_rows=len(history),reward_diagnostics=diagnostics,changed_predictions=changes)))


if __name__=='__main__':main()
