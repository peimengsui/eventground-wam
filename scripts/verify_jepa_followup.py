#!/usr/bin/env python3
"""Independent pilot arithmetic/coverage audit and fresh checkpoint/image replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from run_jepa_followup import assemble
from train_jepa_cpt import configure

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_followup import ResidualJepaPredictor, predict
from eventground_wam.jepa_predictor import JepaPredictor, JepaPredictorConfig, load_jepa_checkpoint
from eventground_wam.pixel_data import FutureInputBuilder, FuturePairDataset
from eventground_wam.pixel_predictor import FrozenQwenFeatures, tensor_state_digest


def resample(groups, values):
    rng = np.random.default_rng(20261002)
    tasks = sorted({g[0] for g in groups})
    point = np.zeros(values.shape[1:]); draws = np.zeros((10000, *values.shape[1:]))
    for task in tasks:
        ids = [i for i, g in enumerate(groups) if g[0] == task]
        sampled = rng.integers(len(ids), size=(10000, len(ids)))
        point += values[ids].mean(0)/len(tasks)
        # Explicit sampled array accumulation, not the runner's weight matrix.
        for column in range(len(ids)):
            draws += values[np.array(ids)[sampled[:, column]]]/(len(ids)*len(tasks))
    return point, draws


def aggregate(rows, fields=('cosine', 'copy')):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['task_key'], row['group_digest']].append(row)
    keys = sorted(grouped)
    values = np.array([[sum(r[f] for r in grouped[k])/len(grouped[k]) for f in fields] for k in keys])
    point, draws = resample(keys, values)
    return point, draws, keys


def select(rows, s):
    return [r for r in rows if r['role'] == s['role'] and (s['camera'] == 'both' or r['camera'] == s['camera'])
            and (s['horizon'] is None or s['horizon'] == r['horizon']
                 or (s['horizon'] == 'medium_long' and r['horizon'] >= 8))
            and (s['motion_bin'] == 'all' or r['motion_bin'] == s['motion_bin'])]


def verify_summaries(rows, summaries):
    differences = []
    for summary in summaries:
        chosen = select(rows, summary); point, draws, groups = aggregate(chosen)
        assert len(chosen) == summary['camera_examples'] and len(groups) == summary['groups']
        assert len({g[0] for g in groups}) == summary['tasks']
        actual = [point[0], point[1], 1-point[0]/point[1],
                  *np.quantile(1-draws[:, 0]/draws[:, 1], [.025, .975])]
        saved = [summary['cosine'], summary['copy'], summary['relative_gain_vs_copy'], *summary['paired_group_bootstrap_95']]
        differences.extend(np.abs(np.array(actual)-saved))
    assert max(differences) < 1e-9
    return float(max(differences))


def independently_collapse(array, groups):
    # The selected panel is balanced: 2 groups in each of 40 tasks.
    assert len(groups) == 80 and len({g[0] for g in groups}) == 40
    result = {}
    for ci, camera in enumerate(CAMERAS):
        x = torch.from_numpy(array[:, ci*64:(ci+1)*64]).double()
        std = float((x-x.mean(0)).square().mean().sqrt())
        pooled = x.mean(1); centered = (pooled-pooled.mean(0))/len(pooled)**.5
        energy = torch.linalg.svdvals(centered).square()
        probability = energy/energy.sum(); probability = probability[probability > 1e-15]
        rank = float(torch.exp(-(probability*probability.log()).sum()))
        unit = pooled/pooled.norm(dim=-1, keepdim=True)
        similarities = []
        for task in sorted({g[0] for g in groups}):
            ix = [i for i, g in enumerate(groups) if g[0] == task]
            assert len(ix) == 2
            similarities.append(float(unit[ix[0]]@unit[ix[1]]))
        result[camera] = {'fixed_position_std_rms': std, 'pooled_effective_rank': rank,
            'within_task_group_cosine': float(np.mean(similarities)),
            'nonzero_pooled_vectors': bool((pooled.norm(dim=-1) > 1e-12).all())}
    return result


def checkpoint(args, arm, contract):
    if arm == 'production_v1_reference':
        model, _ = load_jepa_checkpoint(args.candidate)
        return model
    data = torch.load(args.run/f'{arm}.pt', map_location='cpu', weights_only=True)
    h = data['header']; assert h == seal_contract({k: v for k, v in h.items() if k != 'digest'})
    assert h['contract_digest'] == contract['digest'] and h['updates'] == contract['updates_per_arm']
    assert h['state_sha256'] == tensor_state_digest(data['state_dict'])
    cls = ResidualJepaPredictor if arm == 'copy_residual' else JepaPredictor
    model = cls(JepaPredictorConfig(**h['config'])); model.load_state_dict(data['state_dict'])
    return model


@torch.no_grad()
def replay_probes(model, bank, ids, predictions, tokens, lookup, ablations):
    """Independent NumPy head bound and fresh intervention replay on 40 pairs."""
    x, current, target, _ = assemble(bank, ids[:8])
    maximum_bound = maximum_ablation = 0.
    if not isinstance(model, ResidualJepaPredictor):
        weight = model.output_projection.weight.cpu().numpy().astype(np.float64)
        bias = model.output_projection.bias.cpu().numpy().astype(np.float64)
        matrix = np.column_stack((weight-weight.mean(0), bias-bias.mean()))
        u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        basis = u[:, singular > singular.max()*1e-10]
        for field, values in [('head_span_optimistic_error_bound', target),
                              ('current_head_span_optimistic_error_bound', current)]:
            projected = values.cpu().numpy().astype(np.float64) @ basis
            bounds = np.clip(1-np.linalg.norm(projected, axis=-1), 0, 2).reshape(40, 2, 64).mean(-1)
            for j, i in enumerate(ids[:8]):
                r = bank['records'][i]
                for hi, h in enumerate(HORIZONS):
                    for ci, camera in enumerate(CAMERAS):
                        maximum_bound = max(maximum_bound, abs(bounds[j*5+hi, ci]-lookup[r['episode_index'], r['anchor'], h, camera][field]))
    if ablations is not None:
        original = torch.from_numpy(np.array(predictions[:40])).cuda()
        original_tokens = torch.from_numpy(np.array(tokens[:40])).cuda()
        measured = {(r['kind'], r['episode_index'], r['anchor'], r['horizon'], r['camera']): r for r in ablations}
        # The first eight anchors are two groups of the first task, four each.
        assert len({bank['records'][i]['task_key'] for i in ids[:8]}) == 1
        assert bank['records'][ids[0]]['group_digest'] != bank['records'][ids[4]]['group_digest']
        for kind in ('zero-actions', 'reverse-actions', 'zero-horizon', 'swap-tokens'):
            if kind == 'swap-tokens':
                changed_tokens = torch.cat((original_tokens[20:], original_tokens[:20]))
                raw = model.output_projection(changed_tokens)
                if isinstance(model, ResidualJepaPredictor):
                    raw = current+raw/(2048**.5)
                raw = torch.nn.functional.layer_norm(raw.float(), (2048,))
                features = torch.nn.functional.normalize(raw, dim=-1)
            else:
                value = predict(model, x, current, ablation=kind)
                features, changed_tokens = value.features, value.future_tokens
            cosine = (1-(features.double()*target.double()).sum(-1)).clamp(0, 2).reshape(40, 2, 64).mean(-1)
            feature_change = (features-original).abs().mean((1, 2))
            token_change = (changed_tokens-original_tokens).abs().mean((1, 2))
            for j, i in enumerate(ids[:8]):
                r = bank['records'][i]
                for hi, h in enumerate(HORIZONS):
                    for ci, camera in enumerate(CAMERAS):
                        saved = measured[kind, r['episode_index'], r['anchor'], h, camera]
                        for field, actual in [('cosine', cosine[j*5+hi, ci]),
                            ('mean_absolute_feature_change', feature_change[j*5+hi]),
                            ('mean_absolute_token_change', token_change[j*5+hi])]:
                            maximum_ablation = max(maximum_ablation, abs(float(actual)-saved[field]))
    assert maximum_bound < 1e-8 and maximum_ablation < 1e-6
    return maximum_bound, maximum_ablation


def screen(arm_rows, report, contract):
    control, residual = arm_rows['absolute_control'], arm_rows['copy_residual']
    def cell(rows, camera='both', horizon=None):
        return select(rows, {'role': 'pilot_eval', 'camera': camera, 'horizon': horizon, 'motion_bin': 'all'})
    comparison = []
    for a, b in zip(control, residual, strict=True):
        assert all(a[k] == b[k] for k in ('role', 'task_key', 'group_digest', 'anchor', 'camera', 'horizon'))
        comparison.append({**a, 'control': a['cosine'], 'residual': b['cosine']})
    point, draws, _ = aggregate(cell(comparison), ('control', 'residual'))
    relative = 1-point[1]/point[0]; interval = np.quantile(1-draws[:, 1]/draws[:, 0], [.025, .975])
    overall, _, _ = aggregate(cell(residual)); long, _, _ = aggregate(cell(residual, horizon='medium_long'))
    front, _, _ = aggregate(cell(comparison, CAMERAS[0], 2), ('control', 'residual'))
    criteria = contract['screen_criteria']
    checks = {'front_h2_improvement_vs_control': 1-front[1]/front[0] >= criteria['front_h2_relative_improvement_vs_control_min'],
        'overall_vs_copy': 1-overall[0]/overall[1] >= criteria['overall_relative_gain_vs_copy_min'],
        'medium_long_vs_copy': 1-long[0]/long[1] >= criteria['medium_long_relative_gain_vs_copy_min'],
        'overall_vs_control_uncertainty': interval[0] > criteria['overall_gain_vs_control_bootstrap_lower_bound_gt']}
    cells = []
    for camera in CAMERAS:
        for h in HORIZONS:
            q, _, _ = aggregate(cell(comparison, camera, h), ('control', 'residual', 'copy'))
            cells.append({'camera': camera, 'horizon': h, 'control': q[0], 'residual': q[1], 'copy': q[2],
                          'residual_gain_vs_control': 1-q[1]/q[0], 'residual_gain_vs_copy': 1-q[1]/q[2]})
    checks['all_cells_vs_copy'] = all(c['residual'] <= criteria['all_camera_horizon_max_ratio_to_copy']*c['copy'] for c in cells)
    checks['all_cells_vs_control'] = all(c['residual'] <= criteria['all_camera_horizon_max_ratio_to_control']*c['control'] for c in cells)
    collapse = report['arms']['copy_residual']['collapse']
    for camera in CAMERAS:
        p, t = collapse['prediction'][camera], collapse['target'][camera]
        checks['collapse/'+camera] = bool(p['nonzero_pooled_vectors']
            and p['fixed_position_std_rms'] >= criteria['collapse_std_ratio_min']*t['fixed_position_std_rms']
            and p['pooled_effective_rank'] >= max(criteria['collapse_rank_absolute_min'], criteria['collapse_rank_ratio_min']*t['pooled_effective_rank'])
            and p['within_task_group_cosine'] <= min(criteria['collapse_cosine_absolute_max'], t['within_task_group_cosine']+criteria['collapse_cosine_additive_max']))
    ab = report['arms']['copy_residual']['ablations']
    checks['action_predictive_use'] = ab['zero-actions']['medium_long_cosine_increase'] >= criteria['zero_actions_medium_long_cosine_increase_min']
    checks['tokens_predictive_use'] = ab['swap-tokens']['medium_long_cosine_increase'] >= criteria['swap_tokens_medium_long_cosine_increase_min']
    checks['action_changes_features_and_tokens'] = all(ab['zero-actions'][field] > criteria['action_feature_and_token_change_min']
        for field in ('mean_absolute_feature_change', 'mean_absolute_token_change'))
    return {'status': 'GO_FOR_FURTHER_TRAIN_ONLY_VERIFICATION' if all(checks.values()) else 'NO_GO_OR_INCONCLUSIVE',
        'checks': checks, 'overall_relative_gain_vs_control': float(relative), 'paired_group_bootstrap_95': interval.tolist(),
        'camera_horizon': cells, 'accepted_checkpoint': False, 'production_authorized': False}


@torch.no_grad()
def fresh_images(args, bank, contract):
    root = args.source
    dataset = FuturePairDataset(args.pairs, args.data_root/f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo',
        split='train', expected_manifest_sha256=contract['pair_manifest_sha256'],
        normalizer=load_b0_action_normalizer(root/'docs/model/week4-fresh-v2/action-statistics.json'),
        state_bounds=read_sealed(root/'configs/training/week4_b0_fresh_all_l1_v2.json')['state_normalization'])
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).cuda()
    processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
    encoder = FrozenQwenFeatures(full.model); teacher = FrozenQwenVisualTargets(full.model.visual, processor.image_processor)
    builder = FutureInputBuilder(processor); maximum = 0.
    for first in (0, 8):
        samples = [dataset[dataset.ordinal(bank['records'][i]['episode_index'], bank['records'][i]['anchor'], 2)] for i in range(first, first+4)]
        observations = [s.observation for s in samples]
        batch = builder([s.instruction for s in observations], [s.current_rgb[0] for s in observations], [s.current_rgb[1] for s in observations])
        features, padding = encoder({k: v.cuda() for k, v in batch.items()})
        current = teacher(np.stack([s.current_rgb for s in observations])).cpu()
        for j, i in enumerate(range(first, first+4)):
            assert hashlib.sha256(observations[j].current_rgb.tobytes()).hexdigest() == bank['records'][i]['current_rgb_sha256']
            maximum = max(maximum, float((features[j, ~padding[j]].cpu()-bank['features'][i]).abs().max()),
                          float((current[j]-bank['current'][i]).abs().max()))
            source = dataset.sources[bank['records'][i]['episode_index']]; rgb = dataset._episode(source)[2]
            frames = [bank['records'][i]['anchor']+h for h in HORIZONS]
            targets = teacher(rgb[frames]).cpu()
            maximum = max(maximum, float((targets-bank['targets'][i]).abs().max()))
            assert [hashlib.sha256(rgb[t].tobytes()).hexdigest() for t in frames] == bank['records'][i]['future_rgb_sha256']
    assert maximum == 0
    return {'anchors': 8, 'future_pairs': 40, 'maximum_feature_error': maximum, 'fit_and_pilot_eval_groups': True}


def audit(args):
    configure(); torch.set_num_threads(2)
    root, run = args.source, args.run
    contract = read_sealed(root/'configs/training/jepa_followup_v1.json')
    report = read_sealed(run/'report.json'); complete = read_sealed(run/'completion.json')
    assert complete['report_digest'] == report['digest'] and report['contract_digest'] == contract['digest']
    for name, digest in complete['files_sha256'].items():
        assert sha256_file(run/name) == digest, name
    for name, digest in contract['input_sha256'].items():
        assert sha256_file(root/name) == digest, name
    assert sha256_file(args.candidate) == contract['reference_candidate_sha256']
    bank = torch.load(run/'train-only-bank.pt', map_location='cpu', weights_only=True)
    records = bank['records']; selected = contract['selection']
    assert len(records) == 640
    expected = {(s['episode_index'], t): s for s in selected for t in s['anchors']}
    for r in records:
        s = expected.pop((r['episode_index'], r['anchor']))
        assert r['split'] == s['split'] == 'train' and r['role'] == s['role'] and r['group_digest'] == s['group_digest']
        assert [x['horizon'] for x in r['pair_identities']] == list(HORIZONS)
        assert all(x['target'] == r['anchor']+x['horizon'] and x['action_start'] == r['anchor'] and x['action_stop_exclusive'] == x['target'] for x in r['pair_identities'])
    assert not expected
    fit_ids = [i for i,r in enumerate(records) if r['role'] == 'fit']; eval_ids = [i for i,r in enumerate(records) if r['role'] == 'pilot_eval']
    assert {records[i]['group_digest'] for i in fit_ids}.isdisjoint({records[i]['group_digest'] for i in eval_ids})
    rng = np.random.default_rng(contract['seed']); plan = []
    while len(plan) < contract['updates_per_arm']:
        plan.extend(rng.permutation(320).reshape(-1, 8).tolist())
    plan = plan[:contract['updates_per_arm']]
    assert hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest() == contract['training_plan_sha256']
    chain = '0'*64
    for indices in plan:
        chain = hashlib.sha256((chain+json.dumps([fit_ids[i] for i in indices], separators=(',', ':'))).encode()).hexdigest()
    max_summary = max_scalar = max_replay = max_collapse = max_group_mean = 0.; rows_by_arm = {}
    max_bound = max_ablation = 0.
    for arm in contract['arms']:
        fit = read_sealed(run/f'{arm}-fit-metrics.json')['rows']
        metric = read_sealed(run/f'{arm}-eval-metrics.json'); rows = metric['rows']; rows_by_arm[arm] = fit+rows
        for role, values, ids in [('fit', fit, fit_ids), ('pilot_eval', rows, eval_ids)]:
            identities = {(records[i]['episode_index'], records[i]['anchor'], h, c) for i in ids for h in HORIZONS for c in CAMERAS}
            for r in values:
                identities.remove((r['episode_index'], r['anchor'], r['horizon'], r['camera']))
                assert r['role'] == role
            assert not identities and len(values) == 3200
        max_summary = max(max_summary, verify_summaries(fit+rows, report['arms'][arm]['summaries']))
        lookup = {(r['episode_index'], r['anchor'], r['horizon'], r['camera']): r for r in rows}
        predictions = np.load(run/f'{arm}-eval-predictions.npy', mmap_mode='r')
        tokens = np.load(run/f'{arm}-eval-tokens.npy', mmap_mode='r')
        for j, i in enumerate(eval_ids):
            p = np.asarray(predictions[j*5:(j+1)*5], dtype=np.float64)
            target = bank['targets'][i].numpy().astype(np.float64)
            current = bank['current'][i].numpy().astype(np.float64)
            actual = np.clip(1-np.einsum('ntd,ntd->nt', p, target), 0, 2).reshape(5, 2, 64).mean(-1)
            copied = np.clip(1-np.einsum('td,ntd->nt', current, target), 0, 2).reshape(5, 2, 64).mean(-1)
            for hi,h in enumerate(HORIZONS):
                for ci,c in enumerate(CAMERAS):
                    saved = lookup[records[i]['episode_index'], records[i]['anchor'], h, c]
                    max_scalar = max(max_scalar, abs(saved['cosine']-actual[hi,ci]), abs(saved['copy']-copied[hi,ci]))
                    assert saved['rgb_change'] == float(bank['motion'][i,hi,ci])
        for label in ('prediction', 'target'):
            means = np.load(run/f'{arm}-eval-{label}-group-means.npy')
            groups = [tuple(g) for g in metric['collapse_groups']]
            for gi,key in enumerate(groups):
                positions = [j for j,i in enumerate(eval_ids) if (records[i]['task_key'],records[i]['group_digest']) == key]
                values = [np.array(predictions[j*5:(j+1)*5], dtype=np.float64) if label == 'prediction'
                          else bank['targets'][eval_ids[j]].numpy().astype(np.float64) for j in positions]
                max_group_mean = max(max_group_mean,float(np.max(np.abs(np.concatenate(values).mean(0)-means[gi]))))
            recomputed = independently_collapse(means,groups)
            for camera in CAMERAS:
                for k,v in recomputed[camera].items():
                    max_collapse=max(max_collapse,abs(v-report['arms'][arm]['collapse'][label][camera][k]))
        model = checkpoint(args, arm, contract).eval().cuda()
        for start in (0, len(eval_ids)-8):
            x,current,_,_=assemble(bank,eval_ids[start:start+8])
            with torch.no_grad():p=predict(model,x,current)
            max_replay=max(max_replay,float(np.max(np.abs(p.features.cpu().numpy()-predictions[start*5:(start+8)*5]))),
                           float(np.max(np.abs(p.future_tokens.cpu().numpy()-tokens[start*5:(start+8)*5]))))
        ab = None if arm == 'production_v1_reference' else read_sealed(run/f'{arm}-ablations.json')['rows']
        bound, intervention = replay_probes(model, bank, eval_ids, predictions, tokens, lookup, ab)
        max_bound, max_ablation = max(max_bound, bound), max(max_ablation, intervention)
        del model; torch.cuda.empty_cache()
        if arm != 'production_v1_reference':
            training=report['arms'][arm]['training'];assert training['sample_chain']==chain and training['updates']==6000 and training['pair_presentations']==240000
            ab=read_sealed(run/f'{arm}-ablations.json')['rows'];assert len(ab)==12800
            for kind,saved in report['arms'][arm]['ablations'].items():
                selected_rows=[r for r in ab if r['kind']==kind]
                for r in selected_rows:
                    assert abs(r['normal_cosine']-lookup[r['episode_index'],r['anchor'],r['horizon'],r['camera']]['cosine'])<1e-10
                actual={'medium_long_cosine_increase':sum(r['cosine']-r['normal_cosine'] for r in selected_rows if r['horizon']>=8)/sum(r['horizon']>=8 for r in selected_rows),
                        'mean_absolute_feature_change':sum(r['mean_absolute_feature_change'] for r in selected_rows)/len(selected_rows),
                        'mean_absolute_token_change':sum(r['mean_absolute_token_change'] for r in selected_rows)/len(selected_rows)}
                assert max(abs(actual[k]-saved[k]) for k in actual)<1e-10
    assert max_scalar < 1e-9 and max_summary < 1e-9 and max_replay <= 1e-6 and max_collapse < 1e-8 and max_group_mean < 1e-6
    images=fresh_images(args,bank,contract)
    decision=screen(rows_by_arm,report,contract)
    result = {'artifact_type':'eventground_wam.jepa_followup_independent_verification','status':'passed',
        'contract_digest':contract['digest'],'report_digest':report['digest'],'auditor_sha256':sha256_file(Path(__file__)),
        'all_completion_files_verified':len(complete['files_sha256']),'fit_groups':80,'pilot_eval_groups':80,
        'fit_and_eval_group_disjointness':True,'all_input_splits':'train','matched_pair_presentations_per_arm':240000,
        'matched_sample_chain':chain,'summary_bootstrap_max_error':max_summary,'saved_eval_scalar_max_error':max_scalar,
        'fresh_predictions_and_tokens_max_error':max_replay,'fresh_replay_pairs_per_arm':80,
        'group_mean_max_error':max_group_mean,'independent_collapse_max_error':max_collapse,'fresh_image_replay':images,
        'fresh_ablation_pairs_per_kind_per_fresh_arm':40,'fresh_ablation_max_error':max_ablation,
        'independent_head_bound_pairs_per_absolute_arm':40,'independent_head_bound_max_error':max_bound,
        'pilot_screen':decision,'validation_payload_reads':0,'test_payload_reads':0,'accepted_checkpoint_created':False,
        'limits':'All saved eval errors, aggregate intervals, coverage and artifact hashes checked. Fresh checkpoint replay covers 80 pairs per arm and raw image replay covers 40 pairs; not a full second training run. Gradient-alignment and head-span probes are diagnostics, not causal interventions. The production v1 reference saw all selected train groups.'}
    # Convert NumPy scalar results to the strict contract's native JSON types.
    return seal_contract(json.loads(json.dumps(result, default=lambda value: value.item())))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','run','candidate','pairs','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/home/peimengsui/data/eventground-wam/libero-plus'))
    p.add_argument('--cache',type=Path,default=Path('/home/peimengsui/data/eventground-wam/huggingface-cache'))
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    write_json(args.output,audit(args))
