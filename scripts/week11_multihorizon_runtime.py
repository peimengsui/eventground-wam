"""Input, runtime, budget and endpoint admission for the new continuation."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import check_auxiliary_training as inherited
import torch

from eventground_wam.auxiliary_data import (
    AuxiliaryBatches,
    AuxiliaryPairDataset,
    AuxiliaryTargetCache,
)
from eventground_wam.b0_data import (
    B0DataConfig,
    B0StreamingDataset,
    load_b0_action_normalizer,
    load_trajectory_v2_split_index,
)
from eventground_wam.b0_fresh import load_fresh_recipe
from eventground_wam.b0_model import b0_trainable_state_dict
from eventground_wam.b0_training import (
    atomic_torch_save,
    cpu_tree,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.coupled_checkpoint import export_coupled_policy, load_coupled_policy
from eventground_wam.coupled_policy import require_shared_predictor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.pixel_predictor import tensor_state_digest
from eventground_wam.week11_multihorizon_training import ContinuationEngine, ContinuationPolicy
from eventground_wam.week11_resources import QualificationStopped, ResourceGuard

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/peimengsui/data/eventground-wam/libero-plus')
CACHE = Path('/home/peimengsui/data/eventground-wam/huggingface-cache')
PLAN = ROOT/'docs/evaluation/week11-multihorizon-plan-v1'
CONFIG = ROOT/'configs/training/week11_multihorizon_pilot_v1.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    value = seal_contract(value)
    atomic_json(path, value)
    return value


def emit(path, phase, **fields):
    row = {'phase': phase, 'time_unix': time.time(), **fields}
    atomic_json(path/'progress.json', row)
    print(json.dumps(row), flush=True)


def admission():
    cfg = read_sealed(CONFIG)
    source = read_sealed(ROOT/'week11-multihorizon-source.json')
    require(source['design_digest'] == cfg['digest'], 'source/design mismatch')
    for name, digest in source['files_sha256'].items():
        require(sha256_file(ROOT/name) == digest, 'source changed: '+name)
    for name, digest in cfg['source_files_sha256'].items():
        require(sha256_file(ROOT/name) == digest, 'frozen plan source changed: '+name)
    schedule = read_sealed(PLAN/'training-schedule.json')
    require(schedule['digest'] == cfg['training_schedule']['digest'], 'schedule changed')
    require(batches_digest([r['tokens'] for r in schedule['batches']]) == schedule['action_plan_digest'], 'action order changed')
    return cfg, source, schedule


def runtime_identity():
    import os
    expected = json.loads((ROOT/'launch-evidence/runtime.json').read_text())
    actual = {name: importlib.metadata.version(name) for name in expected['packages']}
    require(actual == expected['packages'], 'training package versions changed')
    require(platform.python_version() == expected['python'] and torch.version.cuda == expected['cuda_build'], 'Python/CUDA changed')
    require(torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 5090', 'wrong training GPU')
    require(os.environ.get('PYTORCH_ALLOC_CONF') == 'expandable_segments:True', 'allocator mismatch')
    return seal_contract({'host': platform.node(), 'python': platform.python_version(), 'packages': actual,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'allocator': os.environ['PYTORCH_ALLOC_CONF'], 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32})


class StageGuard(ResourceGuard):
    def __init__(self, output, identity, cfg, stage):
        self.peak_device_bytes, self.next_device = 0, 0.
        r = cfg['resources']
        hours = {'qualification': r['qualification_hours_total'], 'training': r['training_hours_total'],
                 'screen': r['screen_hours_total']}[stage]
        super().__init__(output, identity, {
            'qualification_active_seconds': hours*3600, 'output_limit_bytes': r[stage+'_output_gib']*2**30,
            'minimum_free_bytes': r['minimum_free_gib']*2**30,
            'host_process_tree_rss_limit_bytes': r['host_process_tree_rss_gib']*2**30,
            'gpu_peak_fraction_strictly_below': r['gpu_fraction_strictly_below']})

    def check(self):
        super().check()
        if time.monotonic() >= self.next_device:
            self.next_device = time.monotonic()+5
            total, used = map(int, subprocess.check_output(['nvidia-smi',
                '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            self.peak_device_bytes = max(self.peak_device_bytes, used*2**20)
            if used >= total*.9:
                raise QualificationStopped('Total device memory including desktop reached 90%')

    def snapshot(self):
        return {**super().snapshot(), 'peak_total_device_bytes': self.peak_device_bytes}


def data_and_models(cfg, schedule):
    started = time.monotonic()
    audit = read_sealed(PLAN/'verified-input-audit.json')
    require(audit['digest'] == cfg['input_audit']['digest'], 'input audit changed')
    for item in audit['input_files'].values():
        require(sha256_file(Path(item['path'])) == item['sha256'], 'input changed: '+item['path'])
    for row in audit['selected_episode_metadata']:
        for item in row['payloads']:
            require(sha256_file(Path(item['path'])) == item['expected_sha256'], 'training payload changed')
    base = load_fresh_recipe(ROOT/'configs/training/week4_b0_fresh_all_l1_v2.json')
    normalizer = load_b0_action_normalizer(ROOT/'docs/model/week4-fresh-v2/action-statistics.json')
    index = load_trajectory_v2_split_index(DATA/'derived/frozen/week2-dataset-v1',
                                         ROOT/'configs/datasets/libero-plus-trajectory-v2')
    repo = DATA/f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo'
    dataset = B0StreamingDataset(DemonstrationReader(repo), index, normalizer, config=B0DataConfig(seed=17))
    pairs = AuxiliaryPairDataset(DATA/'derived/future-pairs/trajectory-v2-future-pairs-v1/primary', repo,
        split='train', expected_manifest_sha256=audit['input_files']['future_pair_manifest']['sha256'],
        normalizer=normalizer, state_bounds=base['state_normalization'], maximum_cache_frames=8192)
    plan = [[tuple(t) for t in r['tokens']] for r in schedule['batches']]
    batches = AuxiliaryBatches(dataset, pairs, plan, CACHE)
    policy, names, predictor, teacher, teacher_hash, _, _ = inherited.models(SimpleNamespace(method='jepa', cache=CACHE), base)
    model = ContinuationPolicy(policy, predictor, normalizer, arm='M4').cuda()
    load_coupled_policy(Path(cfg['starting_endpoints']['m4']['checkpoint_path']), model, names)
    full = torch.load(cfg['full_predictor']['path'], map_location='cpu', weights_only=True)
    require(full['global_step'] == 117132 and full['identity_digest'] == cfg['starting_endpoints']['m4']['identity_digest'], 'full predictor identity changed')
    require(all(torch.equal(v.cpu(), full['state_dict'][k]) for k,v in model.future_core.state_dict().items()), 'full predictor/core mismatch')
    predictor.load_state_dict(full['state_dict'], strict=True)
    del full
    require_shared_predictor(model, predictor)
    require(not any(p.requires_grad for p in teacher.parameters()), 'teacher unfrozen')
    model.train(); predictor.train(); teacher.eval()
    cache = AuxiliaryTargetCache(teacher, maximum_bytes=8*2**30)
    seed_training(17)
    return base, batches, model, predictor, names, teacher, teacher_hash, cache, time.monotonic()-started


def engine_for(values, cfg, source, runtime, arm, mode):
    base, batches, model, predictor, names, *_ = values
    identity = seal_contract({'artifact_type': 'eventground_wam.week11_continuation_identity',
        'design_digest': cfg['digest'], 'source_digest': source['digest'], 'runtime_digest': runtime['digest'],
        'schedule_digest': cfg['training_schedule']['digest'], 'arm': arm, 'mode': mode,
        'starting_policy_sha256': cfg['starting_endpoints']['m4']['policy_sha256'],
        'starting_predictor_sha256': cfg['full_predictor']['sha256']})
    engine = ContinuationEngine(model, names, predictor, base, identity_digest=identity['digest'],
        device=torch.device('cuda'), continuation_arm=arm, design_digest=cfg['digest'], plan=batches.plan)
    require(sum(p.numel() for p in engine.parameters) == cfg['matched']['trainable_parameters_per_arm'], 'trainable inventory changed')
    require(not engine.optimizer.state and engine.scheduler.base_lrs == [1e-5]*4, 'initial optimizer is not fresh')
    seed_training(17)
    return engine, identity


def initial_digest(engine):
    return {k: tensor_state_digest(state) for k,state in {
        'policy': b0_trainable_state_dict(engine.model.policy, allowed_exact_names=engine.exact_lora_names),
        'predictor': engine.predictor.state_dict(), 'coupling': engine.model.coupling.state_dict()}.items()}


def export_endpoint(output, engine, identity, cfg):
    output.mkdir(parents=True, exist_ok=True)
    require(engine.global_step == 2000 and engine.samples_seen == 58172 and engine.future_pairs_seen == 277172, 'endpoint exposure mismatch')
    path = output/'policy.pt'
    metadata = {'continuation_identity': identity['digest'], 'design_digest': cfg['digest'],
                'inherited_updates': 117132, 'continuation_updates': 2000, 'selection': 'fixed-2000-update-endpoint'}
    header = export_coupled_policy(path, engine.model, engine.exact_lora_names, metadata=metadata)
    atomic_torch_save(output/'full-predictor.pt', cpu_tree({'state_dict':engine.predictor.state_dict(), 'identity':identity}))
    return write(output/'endpoint.json', {'artifact_type':'eventground_wam.week11_continuation_endpoint',
        'identity':identity, 'design_digest':cfg['digest'], 'metadata':metadata,
        'policy_sha256':sha256_file(path), 'predictor_sha256':sha256_file(output/'full-predictor.pt'),
        'header_digest':header['digest'], 'state_sha256':header['state_sha256'],
        'samples_seen':engine.samples_seen, 'future_pairs_seen':engine.future_pairs_seen,
        'sample_chain':engine.sample_chain, 'future_chain':engine.future_chain, 'horizon_chain':engine.horizon_chain})


def targets(batch, cache):
    return inherited.get_targets(batch, 'jepa', cache)[0]


def clear(engine):
    engine.model.zero_grad(set_to_none=True)
    engine.predictor.zero_grad(set_to_none=True)
    gc.collect()
