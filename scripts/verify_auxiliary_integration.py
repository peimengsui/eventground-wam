#!/usr/bin/env python3
"""Read-only audit of Week 7 numerical, resource, recovery and source evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]


def verify(attempt):
    run = attempt/'run'
    source = read_sealed(attempt/'source-manifest.json')
    rule = read_sealed(ROOT/'configs/training/week7_auxiliary_pilot_v2.json')
    experiment = read_sealed(ROOT/'configs/training/week7_auxiliary_experiment_v1.json')
    runtime = json.loads((attempt.parent/'runtime.json').read_text())
    assert runtime['host'] == 'feifei' and runtime['baseline_matmul_allow_tf32'] is False
    assert rule['parent_experiment_digest'] == experiment['digest']
    with tarfile.open(attempt/'source.tar.gz', 'r:gz') as archive:
        assert set(archive.getnames()) == {*source['files'], 'source-manifest.json'}
        assert json.load(archive.extractfile('source-manifest.json')) == source
        for name, digest in source['files'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
            assert sha256_file(ROOT/name) == digest, f'current executable/input changed: {name}'
    completion = read_sealed(run/'completion.json')
    assert completion['rule_digest'] == rule['digest']
    assert completion['steps_2_3_4_passed'] is True and completion['production_training_started'] is False
    probes, resumes, calibrations = [], [], []
    limits = rule['correctness']
    for method in rule['methods']:
        report = read_sealed(run/f'{method}-probe-report.json')
        cal = read_sealed(run/f'{method}-calibration.json')
        inventory = read_sealed(run/f'{method}-parameters.json')
        uninterrupted = read_sealed(run/f'{method}-uninterrupted.json')
        resumed = read_sealed(run/f'{method}-resume.json')
        overfit = read_sealed(run/f'{method}-overfit.json')
        for phase in ('probe', 'uninterrupted', 'interrupt', 'resume'):
            preflight = read_sealed(run/f'{method}-{phase}-preflight.json')
            assert preflight['source_digest'] == source['digest']
            assert preflight['rule_digest'] == rule['digest']
            assert preflight['action_samples_per_epoch'] == 1706022
            assert preflight['future_pairs_per_epoch'] == 8129885
            assert preflight['calibration_batches'] == [0, 129, 256, 384]
            assert preflight['validation_payload_reads'] == preflight['test_payload_reads'] == 0
        for value in (report, cal, resumed, uninterrupted):
            assert value['rule_digest'] == rule['digest'] and value['source_digest'] == source['digest']
        assert cal['parameter_inventory_digest'] == inventory['digest']
        assert inventory['future_count'] == (4631427 if method == 'pixel' else 4764928)
        assert inventory['teacher_trainable_count'] == 0
        assert report['calibration_digest'] == cal['digest']
        assert report['overfit_digest'] == overfit['digest']
        ratios = [.1*r['action_hidden_gradient_norm']/r['future_hidden_gradient_norm'] for r in cal['rows']]
        weight = min(10., max(.0001, statistics.median(ratios)))
        assert weight == cal['loss_weight'] == report['calibrated_weight']
        assert cal['optimizer_updates'] == 0 and cal['validation_or_rollout_selection'] is False
        c = report['correctness']
        assert c['future_gradient_comparison']['relative_l2'] <= limits[f'{method}_fp32_gradient_relative_l2']
        assert c['future_gradient_comparison']['maximum_absolute'] <= limits['fp32_gradient_max_absolute']
        assert c['joint_gradient_comparison']['relative_l2'] <= limits['joint_bf16_gradient_relative_l2']
        assert c['loss_difference'] <= limits['full_vs_accumulated_future_loss_atol']
        assert c['auxiliary_gradient_activity']['backbone'] > 0 and c['auxiliary_gradient_activity']['state'] > 0
        assert c['auxiliary_gradient_activity']['head'] == 0 and c['target_gradients'] is False
        assert all(c[k] is True for k in ('action_forward_bitwise_equal', 'future_target_isolation',
                                         'predictor_removal_bitwise_equal'))
        assert all(p['bitwise_equal'] for p in report['parity'])
        assert report['target_repeat_bitwise_equal'] and report['teacher_unchanged']
        assert overfit['passed'] and len(overfit['rows']) == rule['overfit']['updates']
        for loss in ('action', 'future'):
            reduction = 1-overfit['after'][f'{loss}_loss']/overfit['before'][f'{loss}_loss']
            assert reduction >= rule['overfit'][f'minimum_{loss}_loss_relative_reduction']
        assert report['resource_passed'] and len(report['rows']) == rule['resource']['updates']
        assert [r['global_step'] for r in report['rows']] == list(range(1, 129))
        assert report['projected_training_hours'] <= rule['resource']['maximum_projected_hours_per_arm']
        assert report['host_peak_rss_bytes'] <= rule['resource']['maximum_host_rss_gib']*1024**3
        assert report['gpu_peak_reserved_bytes'] <= runtime['cuda_total_bytes']*rule['resource']['maximum_gpu_fraction']
        times = sorted(r['seconds'] for r in report['rows'])
        position = .95*(len(times)-1)
        lower = int(position)
        p95 = times[lower]+(position-lower)*(times[lower+1]-times[lower])
        projected = 1.2*max(statistics.mean(times), p95)*117132/3600
        projected += report['checkpoint_seconds']*59/3600
        assert abs(projected-report['projected_training_hours']) < 1e-8
        assert all(r['finite'] and r['sequence_length'] <= 512 for r in report['edge_probes'])
        assert report['action_only_export_roundtrip_equal']
        assert uninterrupted['state_digest'] == resumed['state_digest']
        assert resumed['process_resume_bitwise_equal'] and resumed['damaged_latest_fallback']
        assert resumed['global_step'] == 4 and resumed['recovery_notes']
        marker = json.loads((run/f'{method}-kill-marker.json').read_text())
        assert marker['global_step'] == 2 and marker['microbatch_stop'] == 10
        assert resumed['sample_chain'] == uninterrupted['sample_chain']
        assert resumed['future_chain'] == uninterrupted['future_chain']
        probes.append(report)
        resumes.append(resumed)
        calibrations.append(cal)
    assert completion['probe_digests'] == [p['digest'] for p in probes]
    assert completion['recovery_digests'] == [r['digest'] for r in resumes]
    for key in ('shared_initialization',):
        assert probes[0][key] == probes[1][key]
    for key in ('sample_chain', 'future_chain', 'samples_seen', 'future_pairs_seen'):
        assert resumes[0][key] == resumes[1][key]
    assert [r['samples_seen'] for r in probes[0]['rows']] == [r['samples_seen'] for r in probes[1]['rows']]
    assert [r['future_pairs_seen'] for r in probes[0]['rows']] == [r['future_pairs_seen'] for r in probes[1]['rows']]
    return seal_contract({'artifact_type': 'eventground_wam.week7_integration_verification',
        'status': 'passed', 'rule_digest': rule['digest'], 'source_digest': source['digest'],
        'completion_digest': completion['digest'], 'verified_source_files': len(source['files']),
        'archive_sha256': sha256_file(attempt/'source.tar.gz'),
        'calibrated_weights': {m: c['loss_weight'] for m, c in zip(rule['methods'], calibrations, strict=True)},
        'shared_initialization': probes[0]['shared_initialization'],
        'evidence_sha256': {str(p.relative_to(run)): sha256_file(p) for p in sorted(run.glob('*.json'))},
        'production_training_started': False,
        'scope': 'Local seals, archived/current source, calibration arithmetic, numerical thresholds, exposure counts, resource gates, and remote process recovery records; remote tensor files remain on Feifei.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path,
                        default=ROOT/'docs/evaluation/week7-integration-v1/attempt-03')
    args = parser.parse_args()
    result = verify(args.attempt)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
