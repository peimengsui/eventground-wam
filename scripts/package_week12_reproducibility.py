"""Package local code, retained analysis inputs and historical reproduction references."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RESULTS = 'docs/evaluation/week12-results-v1'
SCOPE = 'docs/evaluation/week12-scope-v1'
CLOSEOUT = 'docs/evaluation/week11-final-closeout-v1'
FILES = set()
REFERENCES = set()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def add(name):
    p = ROOT / name
    if not p.is_file() or p.is_symlink():
        raise ValueError(f'Missing or non-regular source: {name}')
    FILES.add(name)
    return p


def read(name):
    REFERENCES.add(name)
    return json.loads(add(name).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def collect_code():
    allowed = {'.py', '.sh', '.html', '.json', '.jsonl', '.toml', '.lock', '.conf',
               '.service', '.md', '.txt', '.tsv', '.csv'}
    for folder in ['src', 'scripts', 'tests', 'configs', 'environments', 'docs/model', 'docs/freeze']:
        for p in sorted((ROOT / folder).rglob('*')):
            if any(x in p.parts for x in ('__pycache__', '.venv', '.pytest_cache', '.ruff_cache')):
                continue
            if p.is_file() and p.suffix in allowed:
                add(str(p.relative_to(ROOT)))
    for p in (ROOT / 'docs').glob('*.md'):
        add(str(p.relative_to(ROOT)))
    for name in ['README.md', '.gitignore', '.python-version', 'pyproject.toml', 'uv.lock']:
        add(name)
    for name in [
        'libero-plus-v1/dev-210.json', 'libero-plus-v1/stratified-1K.json', 'libero-plus-v1/sanity-35.json',
        'libero-clean-v1/clean-40.json', 'week4-clean-comparison-v1/paired-2000.json',
        'week11-integration-v1/targeted-467.json', 'week11-integration-v1/clean-200.json',
        'week11-integration-v1/qualification-8.json', 'week11-integration-v1/contact-proxy-98.json']:
        add('evaluation/splits/' + name)


def historical_sources(out):
    w9 = read('outputs/week9-m2-v1-metadata/cohort/cohort.json')
    w11 = read('docs/evaluation/week11-formal-v1/launch/cohort.json')
    analysis = read('docs/evaluation/week11-closeout-v1/remote/analysis.json')
    entries = []
    for label, base, expected, bundled in [
        ('Week 9 runner', 'docs/evaluation/week9-runner-v1/package-04', w9['source_digest'], True),
        ('Week 11 formal runner', 'docs/evaluation/week11-formal-v1/package-v1', w11['source_digest'], False),
        ('Week 11 closeout analyzer', 'docs/evaluation/week11-closeout-v1/package-v3', analysis['analysis_source_digest'], False),
    ]:
        receipt = read(base + '/package.json'); manifest = read(base + '/source-manifest.json')
        archive = ROOT / base / 'source.tar.gz'
        actual = sha(archive)
        if actual != receipt['archive_sha256'] or receipt['source_digest'] != expected or manifest['digest'] != expected:
            raise ValueError(f'Historical source binding mismatch: {label}')
        if bundled:
            add(base + '/source.tar.gz')
        entries.append({'stage': label, 'source_digest': expected, 'archive': base + '/source.tar.gz',
                        'archive_sha256': actual, 'archive_bytes': archive.stat().st_size,
                        'bundled': bundled, 'package_receipt': base + '/package.json',
                        'source_manifest': base + '/source-manifest.json',
                        'disposition': 'Unchanged original archive' if bundled else
                        'Reference only: original archive contains full-10K panel metadata; no panel payload read'})
    write(out / 'historical-sources.json', {'sources': entries, 'current_checkout_is_not_historical_execution_source': True})
    return w9, w11


def checkpoints(out):
    entries = {}
    def record(path, digest, role, source):
        if path in entries:
            if entries[path]['sha256'] != digest:
                raise ValueError(f'Conflicting checkpoint receipts: {path}')
            entries[path]['roles'].append(role); entries[path]['sources'].append(source)
        else:
            entries[path] = {'path': path, 'sha256': digest, 'roles': [role], 'sources': [source],
                             'storage': 'Feifei external payload; not bundled or freshly rehashed in Step 4'}
    admissions = {}
    for arm in ['B0', 'M1', 'M2', 'M3', 'M4']:
        prefix = 'cohort' if arm == 'M2' else 'comparison-context'
        name = f'outputs/week9-m2-v1-metadata/{prefix}/admissions/{arm}.json'
        a = read(name); admissions[arm] = {'path': name, 'record': a}
        record(a['checkpoint_path'], a['checkpoint_sha256'], arm + ' fixed epoch-2 policy', name)
        if arm != 'B0':
            package = 'week9-m2' if arm == 'M2' else 'week9-m4' if arm == 'M4' else 'week9-initial'
            report_name = f'outputs/{package}-v1-metadata/endpoints/{arm}/epoch-02/training-report.json'
            report = read(report_name)
            if report['checkpoint_sha256'] != a['checkpoint_sha256']:
                raise ValueError(f'Endpoint admission mismatch: {arm}')
            filename = 'auxiliary-predictor.pt' if arm in ('M1', 'M2') else 'full-training-predictor.pt'
            record(str(Path(a['checkpoint_path']).parent / filename), report['predictor_sha256'],
                   arm + ' training predictor export', report_name)
    modules_name = 'configs/training/phase2_future_modules_v1.json'
    for name, m in read(modules_name)['modules'].items():
        record(m['remote_path'], m['sha256'], name + '; original gate ' + m['original_gate'], modules_name)
    retained_name = CLOSEOUT + '/retained-checkpoints.json'
    for c in read(retained_name)['checkpoints']:
        record(c['path'], c['sha256'], c['role'], retained_name)
    for tag, arms in [('future-onset', ['sft', 'primary', 'calibration']),
                      ('preservation-pilot', ['hybrid', 'control'])]:
        base = f'docs/evaluation/week10-{tag}-v1'
        protocol = read(base + '/protocol.json')
        package_index = read(base + '/package-index.json')['files_sha256']
        for arm in arms:
            relative = f'runs/{arm}/completion.json'
            name = base + '/' + relative
            completion = read(name)
            evaluation = read(base + f'/eval/dev/{arm}/completion.json')
            receipt = add(base + f'/runs/{arm}/endpoint.pt.sha256')
            if (sha(ROOT / name) != package_index[relative]
                    or completion['digest'] != evaluation['endpoint_completion_digest']
                    or receipt.read_text().strip() != completion['endpoint_sha256']):
                raise ValueError(f'Event checkpoint lineage mismatch: {tag}/{arm}')
            remote = str(Path(protocol['remote_base']) / 'runs' / arm / completion['endpoint_file'])
            record(remote, completion['endpoint_sha256'], 'Week 10 revised task ' + arm, name)
    write(out / 'checkpoints.json', {'checkpoints': list(entries.values()), 'count': len(entries),
          'selection': 'Existing fixed endpoints only; original M4 fixed8 remains Week 11 reference',
          'prior_week11_remote_hash_check': CLOSEOUT + '/remote-closure-check.json',
          'fresh_remote_hash_check_performed': False, 'new_checkpoint_selection': False})
    return admissions


def environments(out, w9, w11):
    names = ['numpy', 'matplotlib', 'Pillow', 'pypdf', 'contourpy', 'cycler', 'fonttools',
             'kiwisolver', 'packaging', 'pyparsing', 'python-dateutil', 'six']
    versions = {n: importlib.metadata.version(n) for n in names}
    (out / 'analysis-requirements.txt').write_text('# Observed Python 3.11 analysis runtime; not the GPU experiment runtime\n' +
                                                '\n'.join(n + '==' + v for n, v in sorted(versions.items())) + '\n')
    locks = []
    for base in ['', 'environments/b0/', 'environments/demonstrations/', 'environments/libero/', 'environments/reference-policy/']:
        project = base + 'pyproject.toml'; lock = base + 'uv.lock'
        metadata = tomllib.loads(add(project).read_text())
        locks.append({'project': project, 'project_sha256': sha(ROOT / project), 'lock': lock,
                      'lock_sha256': sha(add(lock)), 'python': metadata['project']['requires-python'],
                      'declared_dependencies': metadata['project']['dependencies']})
    write(out / 'environments.json', {'analysis': {'python': platform.python_version(),
          'platform': platform.platform(), 'packages': versions, 'new_clean_install_tested': False},
          'locked_projects': locks, 'week9_observed_runtime': w9['runtime'],
          'week11_observed_runtime': w11['runtime'],
          'qualification': 'Observed historical runtime records are authoritative for each run; dependency pins alone do not imply identical renderer behavior.'})


def catalogs(out, admissions):
    model = read('docs/model/Qwen3.5-2B-manifest.json')
    split = read('configs/datasets/libero-plus-trajectory-v2/manifest.json')
    pairs = read('configs/datasets/trajectory-v2-future-pairs-v1/manifest.json')
    order = read('configs/datasets/trajectory-v2-cpt-order-v1/manifest.json')
    external = {
        'backbone': {'repository': model['repository_id'], 'revision': model['revision'],
                     'files': model['files'], 'total_bytes': model['total_size_bytes'],
                     'manifest': 'docs/model/Qwen3.5-2B-manifest.json'},
        'benchmark': {'repository': 'https://github.com/sylvestf/LIBERO-plus.git',
                      'revision': '4976dc30028e805ff8094b55501d532c48fec182',
                      'assets_sha256': '96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf',
                      'setup_reference': 'docs/LIBERO_PLUS_SETUP.md'},
        'demonstrations': {'repository': split['source']['dataset_repo_id'],
                           'revision': split['source']['dataset_revision'],
                           'source_manifest_sha256': split['source']['source_manifest_sha256'],
                           'setup_reference': 'docs/DEMONSTRATIONS.md'},
        'trajectory_split': {'manifest': 'configs/datasets/libero-plus-trajectory-v2/manifest.json', 'digest': split['digest']},
        'future_pairs': {'manifest': 'configs/datasets/trajectory-v2-future-pairs-v1/manifest.json',
                         'digest': pairs['digest'], 'external_outputs': pairs['outputs']},
        'cpt_order': {'manifest': 'configs/datasets/trajectory-v2-cpt-order-v1/manifest.json', 'digest': order['digest']},
        'normalizers': {p: sha(add(p)) for p in ['docs/model/week4-fresh-v2/action-statistics.json',
                                                'docs/model/week4-fresh-v2/state-statistics.json']},
        'payloads_downloaded_or_rehashed_in_step4': False,
        'locked_payloads': 'No full-10K panel or test payload is included or read by the reproduction workflow.'}
    write(out / 'external-assets.json', external)
    rows = []
    def stage(name, entry, configs, evidence, **extra):
        for path in [entry, *configs, *evidence]:
            add(path); REFERENCES.add(path)
        rows.append({'stage': name, 'entrypoint': entry, 'configs': configs, 'evidence': evidence, **extra})
    stage('B0', 'scripts/train_b0_fresh.py', ['configs/training/week4_b0_fresh_all_l1_v2.json'],
          ['docs/evaluation/week4-completion-v1.json', admissions['B0']['path']])
    for kind, week in [('pixel', 5), ('jepa', 6)]:
        stage(kind + ' CPT', f'scripts/train_{kind}_cpt.py', [f'configs/training/week{week}_{kind}_cpt_v1.json'],
              ['configs/training/phase2_future_modules_v1.json'])
    for arm, entry, config in [
        ('M1', 'train_auxiliary_policy.py', 'week7_m1_auxiliary_v1.json'),
        ('M2', 'train_auxiliary_lambda.py', 'week7_m2_auxiliary_v1.json'),
        ('M3', 'train_coupled_policy.py', 'week8_coupled_design_v1.json'),
        ('M4', 'train_coupled_lambda.py', 'week8_coupled_design_v1.json')]:
        a = admissions[arm]
        stage(arm, 'scripts/' + entry, ['configs/training/' + config, 'configs/training/phase2_two_epoch_v1.json'],
              [a['path']], historical_production_source_digest=a['record']['arm_contract']['production_source_digest'],
              historical_training_identity_digest=a['record']['training_identity_digest'])
    stage('Week 9 evaluation', 'scripts/evaluate_week9.py', ['configs/evaluation/week9_factorial_v1.json'],
          ['outputs/week9-m2-v1-metadata/cohort/cohort.json', 'docs/evaluation/week9-final-v1/package-index.json'])
    for tag, entry in [('future-onset', 'train_week10_future_onset.py'), ('preservation-pilot', 'train_week10_preservation.py')]:
        base = f'docs/evaluation/week10-{tag}-v1'
        refs = [base + '/' + name for name in ['package-index.json', 'source-index.json', 'train-manifest.json',
                                               'dev-manifest.json', 'report-manifest.json', 'execution-completion.json']]
        for p in (ROOT / base / 'schedules').glob('*.json'):
            refs.append(str(p.relative_to(ROOT)))
        stage('Week 10 ' + tag, 'scripts/' + entry, [base + '/protocol.json'], refs,
              source_tree=base + '/source', source_tree_included=False)
    stage('Week 11 formal', 'scripts/run_week11_formal.py', ['configs/evaluation/week11_integration_v1.json',
          'configs/evaluation/week11_horizon_mapping_v1.json'], ['docs/evaluation/week11-formal-v1/launch/cohort.json',
          'docs/evaluation/week11-formal-v1/package-v1/package.json', 'docs/evaluation/week11-closeout-v1/package-v3/package.json'])
    stage('Week 11 continuation', 'scripts/run_week11_multihorizon_training.py', ['configs/training/week11_multihorizon_pilot_v1.json'],
          ['docs/evaluation/week11-multihorizon-results-v1/package-index.json'])
    stage('Week 11 adapters', 'scripts/run_week11_motion_training.py', ['configs/training/week11_motion_adapter_v1.json'],
          ['docs/evaluation/week11-motion-adapter-v1/package-index.json', 'docs/evaluation/week11-motion-screen-v1/package-index.json'])
    write(out / 'run-catalog.json', {'stages': rows, 'historical_commands_executed_in_step4': False,
          'instructions': 'Use the guide and exact historical source/runtime. New training or rollouts require separate scope and new identities; do not bypass host/output checks.'})
    ledger = read(CLOSEOUT + '/evidence-ledger.json')
    outputs = []
    # The ledger uses package-specific schemas; preserve each original index verbatim.
    packages = ledger['packages']
    for name, record in packages.items():
        index = record['index']; add(index)
        if sha(ROOT / index) != record['index_sha256']:
            raise ValueError(f'Prior evidence index changed: {name}')
        outputs.append({'name': name, **record, 'payload_scope': 'Index/reference bundled; complete historical payload remains at original locations'})
    for base in ['docs/evaluation/week9-final-v1', 'docs/evaluation/week10-closeout-v1', RESULTS]:
        index = base + '/package-index.json'; add(index)
        outputs.append({'name': Path(base).name, 'index': index, 'index_sha256': sha(ROOT / index)})
    raw_name = 'docs/evaluation/week11-closeout-v1/remote/raw-inventory.json'; raw = read(raw_name)
    write(out / 'outputs.json', {'packages': outputs, 'formal_raw_inventory': raw_name,
          'formal_raw_root': raw['root'], 'formal_raw_files': len(raw['files']),
          'prior_remote_closure_receipt': CLOSEOUT + '/remote-closure-check.json',
          'week9_per_episode_paths_and_hashes': 'The ten bundled evaluation-report.json files retain original per-episode artifact/audit receipts.',
          'preexisting_mismatches': ledger['preexisting_mismatches'],
          'remote_payloads_rehashed_in_step4': False})
    for name in ['handoff.json', 'retained-checkpoints.json', 'remote-closure-check.json', 'package-index.json']:
        add(CLOSEOUT + '/' + name)
    add('docs/evaluation/week10-closeout-v1/handoff.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--git', default='git')
    args = parser.parse_args(); out = args.output.resolve()
    if (out / 'package-index.json').exists():
        raise ValueError('Refusing to rewrite a sealed package')
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'GUIDE.md').is_file() or not (out / 'README.md').is_file():
        raise ValueError('Prepare the guide and README before packaging')
    collect_code()
    for name, expected in read(RESULTS + '/source-manifest.json')['files_sha256'].items():
        if sha(add(name)) != expected:
            raise ValueError(f'Frozen analysis input changed: {name}')
    for base in [SCOPE, RESULTS]:
        index = read(base + '/package-index.json')
        for name, expected in index['files_sha256'].items():
            if sha(add(base + '/' + name)) != expected:
                raise ValueError(f'Sealed package changed: {base}/{name}')
    w9, w11 = historical_sources(out)
    admissions = checkpoints(out)
    environments(out, w9, w11)
    catalogs(out, admissions)
    git = lambda *parts: subprocess.check_output([args.git, *parts], cwd=ROOT, text=True).strip()
    status = git('status', '--porcelain=v1', '--untracked-files=normal')
    write(out / 'repository-state.json', {'head': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
          'dirty': bool(status), 'porcelain_status': status.splitlines(),
          'identity_rule': 'Bundle file hashes identify the captured working tree. HEAD alone does not identify completed work.',
          'base_history_modified': False})
    (out / 'verify.py').write_bytes((ROOT / 'scripts/verify_week12_reproducibility.py').read_bytes())
    for name in ['README.md', 'GUIDE.md', 'analysis-requirements.txt', 'checkpoints.json', 'external-assets.json',
                 'environments.json', 'run-catalog.json', 'outputs.json', 'historical-sources.json', 'verify.py']:
        add(str((out / name).relative_to(ROOT)))
    entries = {}
    compiled = 0
    for name in sorted(FILES):
        path = ROOT / name
        entries[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        if path.suffix == '.py' and name.split('/')[0] in ('scripts', 'src', 'tests'):
            ast.parse(path.read_text(), filename=name); compiled += 1
    if any('full-10K.json' in p or p.endswith('/test.bin') for p in entries):
        raise ValueError('Locked payload in archive inventory')
    archive = out / 'analysis-reproduction.tar.gz'
    with archive.open('wb') as target, gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for name in entries:
                path = ROOT / name; info = tar.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    # Check that no captured file moved during the build.
    for name, record in entries.items():
        if sha(ROOT / name) != record['sha256']:
            raise ValueError(f'Source changed during packaging: {name}')
    write(out / 'bundle-manifest.json', {'artifact_type': 'eventground_wam.week12_reproduction_bundle',
          'archive': archive.name, 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
          'file_count': len(entries), 'uncompressed_bytes': sum(r['bytes'] for r in entries.values()),
          'files': entries, 'python_sources_parsed': compiled,
          'categories': dict(Counter(name.split('/')[0] for name in entries)),
          'referenced_inputs': sorted(REFERENCES), 'new_model_calls': 0, 'new_rollouts': 0,
          'new_training_updates': 0, 'new_remote_connections': 0})
    print(json.dumps({'status': 'built', 'files': len(entries), 'python_sources_parsed': compiled,
                      'archive_bytes': archive.stat().st_size, 'output': str(out)}))


if __name__ == '__main__':
    main()
