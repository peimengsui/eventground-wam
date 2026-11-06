# Week 11 Step 1 freeze

The [experiment](../../WEEK11_EXPERIMENT.md) and
[machine contract](../../../configs/evaluation/week11_integration_v1.json)
freeze the selected SFT-only endpoint, original M4, three arms, exact panel
membership, resource caps and decision criteria. This is a scientific design
package; integration code and an executable rollout launcher are not included.

Run from the repository root:

```bash
.venv/bin/python scripts/verify_week11_protocol.py
.venv/bin/python -m pytest tests/test_week11_protocol.py -q
```

`verification.json` records local seal, provenance, parent cross-link, selector,
overlap and episode-accounting verification. The six focused tests include
rejection of resealed threshold/episode changes, changed endpoint provenance
and a false claim of fresh remote verification. `validation.json` records the
checks performed for this freeze.

`entry.json` pins the evidence available at design time. Its original-evidence
hashes remain unchanged. `before/` preserves the prior navigation documents;
`after/` preserves their Step 1 updates and the new status ledger. The package
index hashes these records and the repository-relative deliverables. Parent
artifacts remain at their pinned repository/Feifei locations; this package is
not a self-contained model deployment.

The contract digest is
`5d7e919252104f9814086474f6a92633edc1d86bff984608c0bbb80a7b4c4a0a`.
Mapping and fallback choices belong to Step 2. The actual checkpoint bytes
were last verified remotely on October 24; Step 1 only cross-checks that record.
No remote deployment, model inference, training or robot rollout occurred.
