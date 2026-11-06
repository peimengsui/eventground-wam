"""Measure a frozen training-only gradient ratio, with no optimizer updates."""
import argparse
import copy
import json
import signal
import socket
import time
from pathlib import Path

import torch

from train_week10_eventground import check_gpu_exclusive, require_output_capacity, write_completion
from eventground_wam.b0_training import configure_deterministic_cuda, seed_training
from eventground_wam.eventground_data import load_current_observation, collate_current_observations, target_tokens
from eventground_wam.eventground_future_onset import audit_row, weak_reward
from eventground_wam.eventground_future_runtime import load_endpoint
from eventground_wam.eventground_model import serialize_tokens
from eventground_wam.eventground_preservation import PreservationTrainer, preservation_weight
from eventground_wam.eventground_preservation_runtime import verify, parent_base
from eventground_wam.eventground_runtime import build_eventground_from_m4, verify_eventground_parent
from eventground_wam.eventground_weak_supervision import sealed, sha
from eventground_wam.pixel_predictor import tensor_state_digest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, required=True)
    base = parser.parse_args().base; p, index, manifests, recipe = verify(base)
    if socket.gethostname() != 'feifei' or str(base.resolve()) != p['remote_base']:
        raise ValueError('Host/base changed')
    schedule = sealed(base/'diagnostic-schedule.json'); root = base/'source'
    resources = p['diagnostic_resources']; check_gpu_exclusive()
    out = base/'diagnostic'; out.mkdir(exist_ok=False); start = time.monotonic()
    report = dict(artifact_type='eventground_wam.preservation_weight_diagnostic', status='failed_preserved',
                  protocol_digest=p['digest'], schedule_digest=schedule['digest'], optimizer_updates=0,
                  dev_examples_scored=0, dev_metrics_used_for_weight=False, automatic_retries=0)
    def budget():
        if time.monotonic()-start > resources['wall_seconds']:
            raise TimeoutError('Diagnostic wall ceiling')
        require_output_capacity(out, resources, 0)
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > .9*torch.cuda.get_device_properties(0).total_memory:
            raise RuntimeError('Diagnostic GPU ceiling')
    def stop(*_):
        raise TimeoutError('Diagnostic stopped; no automatic retry')
    signal.signal(signal.SIGALRM, stop); signal.signal(signal.SIGTERM, stop); signal.alarm(resources['wall_seconds'])
    try:
        configure_deterministic_cuda(); torch.set_num_threads(4); seed_training(17)
        torch.cuda.set_per_process_memory_fraction(.9); torch.cuda.reset_peak_memory_stats()
        device = torch.device('cuda:0'); verified = {}
        rows = {r['prompt_id']: r for r in manifests['train']['rows']}
        candidates = {r['candidate']['recording_id']: r['candidate'] for r in index['panels']['train']}
        observations = {}
        for pid in [pid for batch in schedule['batches'] for pid in batch]:
            row = rows[pid]; audit_row(row, candidates[row['recording_id']])
            observations[pid] = load_current_observation(row, verified_files=verified); budget()
        model, builder, allowlist, parent = build_eventground_from_m4(root, recipe, Path(p['cache']), device)
        if {k:list(v) for k,v in allowlist.items()} != recipe['exact_allowlist']:
            raise ValueError('Allowlist changed')
        completed = load_endpoint(model, allowlist, parent_base(base), 'sft')
        reference = copy.deepcopy(model); reference.requires_grad_(False); reference.eval()
        trainer = PreservationTrainer(model, allowlist, recipe)
        initial = tensor_state_digest(dict(model.named_parameters()))
        reference_initial = tensor_state_digest(dict(reference.named_parameters()))
        if initial != reference_initial:
            raise ValueError('Reference differs before diagnostic')
        groups = []
        def reward(evidence, tokens):
            predictions = serialize_tokens(tokens); values = []
            for i,row in enumerate(evidence):
                group = predictions[4*i:4*i+4]
                scores = [weak_reward(x, row, candidates[row['recording_id']], p['reward_weights']['primary']) for x in group]
                values.append(scores); groups.append(dict(prompt_id=row['prompt_id'], predictions=group, rewards=scores))
            return torch.tensor(values, dtype=torch.float32)
        measurements = []; grpo_norms = []; sft_norms = []
        for batch in schedule['batches']:
            micros = []
            for pid in batch:
                encoded, state = collate_current_observations([observations[pid]], builder, device)
                micros.append((encoded, state, [rows[pid]], target_tokens([rows[pid]], device)))
            grpo = trainer.gradient_diagnostic(micros, prompt_count=len(batch), mode='rlvr', reference_model=reference, reward_function=reward)
            supervised = trainer.gradient_diagnostic(micros, prompt_count=len(batch), mode='control')
            grpo_norms.append(grpo['gradient_norm']); sft_norms.append(supervised['gradient_norm'])
            measurements.append(dict(prompt_ids=batch, grpo=grpo, sft=supervised)); budget()
        if tensor_state_digest(dict(model.named_parameters())) != initial or tensor_state_digest(dict(reference.named_parameters())) != reference_initial:
            raise ValueError('Diagnostic changed parameters')
        if trainer.optimizer.state or trainer.global_step or trainer.prompts_seen or trainer._in_update:
            raise ValueError('Diagnostic changed optimizer')
        for path, (_, fingerprint) in verified.items():
            st = Path(path).stat()
            if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != fingerprint:
                raise ValueError('Training input changed')
        verify_eventground_parent(sealed(root/'configs/training/week10_eventground_experiment_v3.json'))
        if sha(parent_base(base)/'runs/sft/endpoint.pt') != completed['endpoint_sha256']:
            raise ValueError('Stored SFT changed')
        report.update(status='complete', selected_weight=preservation_weight(grpo_norms, sft_norms),
                      grpo_gradient_norms=grpo_norms, sft_gradient_norms=sft_norms, measurements=measurements,
                      sft_completion_digest=completed['digest'], initial_all_parameter_sha256=initial,
                      source_files_verified=len(verified), groups=groups,
                      gates=dict(parameters_unchanged=True, optimizer_empty=True, reference_unchanged=True,
                                 current_observation_only=True, training_targets_reaudited=True, stored_parents_unchanged=True,
                                 training_inputs_unchanged=True, fixed_training_only_rule=True))
        budget()
    except Exception as error:
        report['error'] = repr(error); raise
    finally:
        signal.alarm(0); report['elapsed_seconds'] = time.monotonic()-start
        if torch.cuda.is_initialized():
            report.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        write_completion(out, report, resources)
        print(json.dumps({k:report.get(k) for k in ('status','selected_weight','grpo_gradient_norms','sft_gradient_norms','elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
