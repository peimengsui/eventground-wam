import copy
import json

import pytest
import torch

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_preservation import PreservationTrainer, preservation_weight
from eventground_wam.eventground_preservation_runtime import selected_weight
from test_eventground_training import setup, batches, state_copy, reference_copy, reward_function


def trainer():
    old = setup("rlvr")
    return PreservationTrainer(old.model, old.allowlist, old.recipe)


def inputs(sizes=(5,)):
    return [(enc, state, evidence, target[2]) for (enc, state, evidence), target
            in zip(batches(sizes, rl=True), batches(sizes), strict=True)]


def test_weight_rule_and_fail_closed_inputs():
    assert preservation_weight([2., 4.], [3., 7.]) == pytest.approx(.15)
    assert preservation_weight([1e5, 1e5], [1., 1.]) == 1.
    assert preservation_weight([1e-5, 1e-5], [1., 1.]) == .01
    for a, b in [([1.], [1.]), ([0., 1.], [1., 1.]), ([float("nan"), 1.], [1., 1.])]:
        with pytest.raises(ValueError):
            preservation_weight(a, b)


def test_zero_weight_matches_original_rlvr():
    old, new = setup("rlvr"), trainer()
    torch.manual_seed(109)
    original = old.rlvr_update(batches(rl=True), prompt_count=5, reference_model=reference_copy(old),
                               reward_function=reward_function)
    torch.manual_seed(109)
    revised = new.update(inputs(), prompt_count=5, mode="hybrid", weight=0.,
                         reference_model=reference_copy(new), reward_function=reward_function)
    assert original["loss"] == pytest.approx(revised["loss"], abs=1e-8)
    for n, value in old.model.state_dict().items():
        torch.testing.assert_close(value, new.model.state_dict()[n], atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("mode", ["hybrid", "control"])
def test_unequal_partial_batch_equivalence_and_frozen_reference(mode):
    whole, micro = trainer(), trainer()
    ref1, ref2 = reference_copy(whole), reference_copy(micro)
    before = state_copy(whole)
    torch.manual_seed(109)
    a = whole.update(inputs(), prompt_count=5, mode=mode, weight=.15, reference_model=ref1, reward_function=reward_function)
    torch.manual_seed(109)
    b = micro.update(inputs((2, 3)), prompt_count=5, mode=mode, weight=.15, reference_model=ref2, reward_function=reward_function)
    assert a["loss"] == pytest.approx(b["loss"], abs=2e-7)
    assert a["prompts_seen"] == b["prompts_seen"] == 5
    for n, value in whole.model.state_dict().items():
        torch.testing.assert_close(value, micro.model.state_dict()[n], atol=2e-7, rtol=3e-5)
        if n not in whole.names:
            assert torch.equal(value, before[n])
        assert torch.equal(ref1.state_dict()[n], before[n])
    assert [g["lr"] for g in whole.optimizer.param_groups] == [1e-5, 1e-6]


def test_tied_rewards_still_supply_supervised_gradient():
    t = trainer()
    result = t.update(inputs(), prompt_count=5, mode="hybrid", weight=.15, reference_model=reference_copy(t),
                       reward_function=lambda e, tokens: torch.ones(len(e), 4))
    assert result["equal_reward_groups"] == 5
    assert abs(result["policy_loss"]) < 1e-8
    assert result["gradient_norm"] > 0
    assert result["loss"] == pytest.approx(.15 * result["sft_loss"], abs=1e-7)


def test_diagnostic_never_updates_parameters_or_optimizer():
    t = trainer(); before = state_copy(t); ref = reference_copy(t)
    for mode in ("rlvr", "control"):
        result = t.gradient_diagnostic(inputs(), prompt_count=5, mode=mode, reference_model=ref, reward_function=reward_function)
        assert result["gradient_norm"] > 0 and result["optimizer_updates"] == 0
        assert not t.optimizer.state and not t._in_update and t.global_step == t.prompts_seen == 0
        assert all(torch.equal(v, before[n]) for n, v in t.model.state_dict().items())
        assert all(p.grad is None for p in t.model.parameters())


def test_hybrid_recovery_reproduces_candidate_sampling_and_update(tmp_path):
    t = trainer(); ref = reference_copy(t)
    kw = dict(prompt_count=5, mode="hybrid", weight=.15, reference_model=ref, reward_function=reward_function)
    t.update(inputs(), **kw)
    identity = seal_contract(dict(experiment="synthetic-preservation", mode="hybrid", weight=.15))
    path = tmp_path / "recovery.pt"; t.save(path, identity=identity)
    expected = t.update(inputs(), **kw)
    resumed = trainer(); resumed.resume(path, identity=identity)
    actual = resumed.update(inputs(), **kw)
    assert actual == expected
    assert all(torch.equal(v, resumed.model.state_dict()[n]) for n, v in t.model.state_dict().items())


def test_shared_reference_and_wrong_target_count_rejected():
    t = trainer()
    with pytest.raises(ValueError, match="reference"):
        t.update(inputs(), prompt_count=5, mode="hybrid", weight=.15, reference_model=t.model)
    bad = inputs(); enc, state, evidence, tokens = bad[0]
    with pytest.raises(ValueError, match="Unpaired"):
        t.update([(enc, state, evidence, tokens[:2])], prompt_count=5, mode="control")


def test_diagnostic_admission_rejects_mutated_rule_or_dev_use(tmp_path):
    p = seal_contract(dict(diagnostic_resources=dict(wall_seconds=600), sft_completion_digest='a'*64))
    s = seal_contract(dict(batches=[list(range(16)), list(range(16,32))]))
    def save(name, value):
        (tmp_path/name).write_text(json.dumps(value if 'digest' in value else seal_contract(value)))
    save('protocol.json', p); save('diagnostic-schedule.json', s); (tmp_path/'diagnostic').mkdir()
    d = dict(status='complete', protocol_digest=p['digest'], schedule_digest=s['digest'],
             sft_completion_digest='a'*64, optimizer_updates=0, dev_examples_scored=0, dev_metrics_used_for_weight=False,
             gates=dict(unchanged=True), elapsed_seconds=40, peak_reserved_bytes=20, gpu_total_bytes=32,
             selected_weight=.15, grpo_gradient_norms=[2.,4.], sft_gradient_norms=[3.,7.])
    save('diagnostic/completion.json', d)
    assert selected_weight(tmp_path)[0] == pytest.approx(.15)
    for key, value in [('selected_weight',.8), ('optimizer_updates',1), ('dev_examples_scored',1),
                       ('dev_metrics_used_for_weight',True), ('gates',dict(unchanged=False))]:
        save('diagnostic/completion.json', dict(d, **{key:value}))
        with pytest.raises(ValueError):
            selected_weight(tmp_path)
