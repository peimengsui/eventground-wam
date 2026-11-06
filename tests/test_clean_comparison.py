from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_episode_recorder import step_record

from eventground_wam.clean_acceptance import read_sealed
from eventground_wam.clean_comparison import (
    ComparisonIntegrityError,
    compare_pair_receipts,
    execute_pair,
    metadata_for_pair,
    paths_for_pair,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import EpisodeRecorder, EpisodeValidationError
from eventground_wam.evaluation_protocol import derive_episode_seed, load_evaluation_protocol

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def case(tmp_path):
    protocol = load_evaluation_protocol(ROOT / 'configs/evaluation/week4_clean_comparison_v1.json')
    panel = read_sealed(ROOT / 'evaluation/splits/week4-clean-comparison-v1/paired-2000.json')
    rule = read_sealed(ROOT / 'configs/evaluation/week4_clean_acceptance_v1.json')
    pair = panel['pairs'][1]
    identity = seal_contract({'project_git_revision': 'a' * 40,
                             'checkpoint_id': 'test', 'checkpoint_revision': 'v1',
                             'acceptance_rule_digest': rule['digest']})
    seeds = []
    return {'root': tmp_path, 'ordinal': 1, 'pair': pair, 'identity': identity, 'protocol': protocol,
                'metadata': metadata_for_pair(pair, protocol, identity, Path('/test/task.bddl')),
                'initial_state': np.zeros(12), 'policy': SimpleNamespace(
                    predict_chunk=lambda **kwargs: np.zeros((8, 7))),
                'environment_factory': lambda: SimpleNamespace(close=lambda: None),
                'episode_runner': write_episode, 'seed_episode': seeds.append}


def write_episode(*, protocol, metadata, artifact_path, policy, **kwargs):
    with EpisodeRecorder(artifact_path, protocol, metadata) as recorder:
        for i in range(10):
            recorder.append(step_record(protocol, i, policy=False))
        record = step_record(protocol, 10, policy=True, done=True)
        try:
            policy.predict_chunk(agent_rgb=record.agent_rgb, wrist_rgb=record.wrist_rgb,
                                 proprioception=record.proprioception)
        except RuntimeError:
            recorder.finalize(success=False, termination_reason='policy_exception:RuntimeError')
            return
        recorder.append(record)
        recorder.finalize(success=True, termination_reason='environment_done')


def test_all_panel_trial_seeds_and_trial_metadata(case):
    panel = read_sealed(ROOT / 'evaluation/splits/week4-clean-comparison-v1/paired-2000.json')
    for p in panel['pairs']:
        assert p['episode_seed'] == derive_episode_seed(
            case['protocol'], p['suite'], p['runtime_task_index'], p['trial_index'])
    with pytest.raises(EpisodeValidationError):
        EpisodeRecorder(case['root'] / 'wrong.hdf5', case['protocol'],
                        replace(case['metadata'], trial_index=0))


def test_completed_policy_failure_never_retried(case):
    def fail(**kwargs):
        raise RuntimeError('policy failure')
    case['policy'] = SimpleNamespace(predict_chunk=fail)
    first = execute_pair(**case)
    assert not first['success'] and first['assessable']
    assert first['first_query_input_hashes'] is not None
    assert execute_pair(**case) == first
    assert len(case['seed_episode'].__self__) == 1


def test_interruption_before_simulator_consumes_attempt_and_reuses_seed(case):
    def interrupt():
        raise KeyboardInterrupt
    factory = case['environment_factory']
    case['environment_factory'] = interrupt
    with pytest.raises(KeyboardInterrupt):
        execute_pair(**case)
    case['environment_factory'] = factory
    result = execute_pair(**case)
    assert result['attempt_index'] == 1
    assert result['success']
    assert case['seed_episode'].__self__ == [case['pair']['episode_seed']] * 2
    assert execute_pair(**case) == result


def test_partial_preserved_and_single_retry(case):
    def interrupt(**kwargs):
        with EpisodeRecorder(kwargs['artifact_path'], kwargs['protocol'], kwargs['metadata']) as rec:
            rec.append(step_record(kwargs['protocol'], 0, policy=False))
            raise KeyboardInterrupt
    case['episode_runner'] = interrupt
    with pytest.raises(KeyboardInterrupt):
        execute_pair(**case)
    artifact = paths_for_pair(case['root'], 1, case['pair'])[0]
    assert Path(str(artifact) + '.partial').exists()
    case['episode_runner'] = write_episode
    result = execute_pair(**case)
    assert result['attempt_index'] == 1
    assert result['attempt_evidence'][0]['partial'] is not None
    assert execute_pair(**case) == result


def test_completed_artifact_recovered_after_process_interrupt(case):
    def interrupt(**kwargs):
        write_episode(**kwargs)
        raise KeyboardInterrupt
    case['episode_runner'] = interrupt
    with pytest.raises(KeyboardInterrupt):
        execute_pair(**case)
    case['episode_runner'] = lambda **kwargs: pytest.fail('completed outcome reran')
    result = execute_pair(**case)
    assert result['success'] and result['attempt_index'] == 0
    assert len(case['seed_episode'].__self__) == 1


def test_exhausted_infrastructure_kept_in_denominator(case):
    def unavailable():
        raise OSError('simulator unavailable')
    case['environment_factory'] = unavailable
    result = execute_pair(**case)
    assert result['success'] is False and result['assessable'] is False
    assert result['termination_reason'] == 'infrastructure_failure'
    assert result['attempt_index'] == 1
    assert execute_pair(**case) == result
    assert len(case['seed_episode'].__self__) == 2


def test_receipt_detects_artifact_tampering(case):
    result = execute_pair(**case)
    with Path(result['artifact']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ComparisonIntegrityError):
        execute_pair(**case)
    assert len(case['seed_episode'].__self__) == 1


def test_input_mismatch_blocks_scientific_validity(case):
    left = execute_pair(**case)
    right = dict(left, first_query_input_hashes={'agent_rgb': 'wrong'})
    assert not compare_pair_receipts(left, right)['assessable']
    assert compare_pair_receipts(left, left)['assessable']
