"""Reuse byte-identical, previously verified paired rollout evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eventground_wam.clean_acceptance import analyze_paired_outcomes, read_sealed
from eventground_wam.clean_comparison import (
    ComparisonIntegrityError,
    compare_pair_receipts,
    load_pair_receipt,
    paths_for_pair,
)
from eventground_wam.episode_recorder import sha256_file


def verify_cached_record(root: Path, ordinal: int, pair: Any, pinned: Any,
                         policy: str, identity: Any) -> dict[str, Any]:
    """Bind bytes to the old final audit; do not rewrite the original identity.

    The pinned receipt was structurally validated by the original completed
    comparison. Hashing its artifacts and journals again establishes that those
    exact verified bytes are still present without decompressing every frame.
    """
    record = load_pair_receipt(root, ordinal, pair)
    if (record['digest'] != pinned[f'{policy}_receipt_digest']
            or record['pair'] != pair or record['ordinal'] != ordinal
            or record['identity_digest'] != identity['digest']
            or record['pair_id'] != pinned['pair_id']
            or record['success'] != pinned[f'{policy}_success']
            or record['assessable'] is not True or not pinned['assessable']):
        raise ComparisonIntegrityError('cached receipt differs from the verified original panel')
    artifact, _, _ = paths_for_pair(root, ordinal, pair, record['attempt_index'])
    if record['artifact'] != str(artifact) or sha256_file(artifact) != record['artifact_sha256']:
        raise ComparisonIntegrityError('cached artifact bytes changed')
    for attempt in record['attempt_evidence']:
        expected_artifact, expected_journal, _ = paths_for_pair(
            root, ordinal, pair, attempt['attempt_index'])
        for name, path in [('journal', expected_journal),
                           ('partial', Path(str(expected_artifact) + '.partial'))]:
            evidence = attempt[name]
            if evidence is not None and (
                    evidence['path'] != str(path) or sha256_file(path) != evidence['sha256']):
                raise ComparisonIntegrityError('cached attempt evidence changed')
    return record


def audit_cached_comparison(root: Path, config: Any, panel: Any,
                            on_progress: Any = None) -> tuple[Any, Any]:
    final = read_sealed(root / 'comparison-report.json')
    if (final['digest'] != config['reuse']['comparison_report_digest']
            or final['panel_digest'] != panel['digest']
            or final['pair_count'] != panel['pair_count']
            or len(final['pairs']) != panel['pair_count']
            or not final['all_pairs_valid'] or final['mode'] != 'formal'):
        raise ComparisonIntegrityError('cached final comparison differs from the frozen source')
    records = {}
    for policy in ('b0', 'reference'):
        identity = read_sealed(root / policy / 'identity.json')
        if identity != final['policy_identities'][policy]:
            raise ComparisonIntegrityError('cached policy identity changed')
        records[policy] = []
        for ordinal, pair in enumerate(panel['pairs']):
            record = verify_cached_record(root / policy, ordinal, pair,
                                          final['pairs'][ordinal], policy, identity)
            records[policy].append(record)
            if on_progress is not None and (ordinal + 1) % 100 == 0:
                on_progress(policy, ordinal + 1)
        report = read_sealed(root / policy / 'evaluation-report.json')
        if (report['status'] != 'complete' or report['identity_digest'] != identity['digest']
                or report['receipt_digests'] != [r['digest'] for r in records[policy]]):
            raise ComparisonIntegrityError('cached policy completion report changed')
    checks = [compare_pair_receipts(a, b) for a, b in zip(
        records['b0'], records['reference'], strict=True)]
    if checks != final['pairs']:
        raise ComparisonIntegrityError('cached paired outcomes or input parity changed')
    return final, records


def comparison_statistics(new: list[Any], previous: list[Any], *, gate: bool) -> dict[str, Any]:
    result = analyze_paired_outcomes(*[
        np.asarray([r['success'] for r in rows], dtype=bool).reshape(4, 10, 50)
        for rows in (new, previous)])
    if not gate:
        # The old B0 is a descriptive improvement comparison, not the gate reference.
        result.pop('statistical_decision')
        result.pop('acceptance_rule_id')
    return result
