from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eventground_wam.b0_training import seed_training
from eventground_wam.jepa_predictor import (
    JepaPredictor,
    JepaPredictorConfig,
    jepa_prediction_loss,
    normalize_predictions,
)
from eventground_wam.jepa_training import (
    LOSS_RULE,
    TARGET_BYTES,
    JepaTrainingEngine,
    PoolTargetCache,
    accumulate_loss,
    anchor_groups,
)
from eventground_wam.pixel_predictor import PixelCondition
from eventground_wam.pixel_training import PixelCheckpointStore, tree_digest


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model():
    return JepaPredictor(JepaPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1))


def condition(n):
    return PixelCondition(torch.randn(n, 3, 8), torch.zeros(n, 3, dtype=torch.bool),
                          torch.rand(n, 2, 3, 128, 128), torch.randn(n, 8),
                          torch.randn(n, 32, 7), torch.tensor([8]*n))


@pytest.mark.parametrize("groups,microbatch", [([0, 0, 0, 1, 1, 2], 2), ([0, 0, 1, 1, 2], 3), ([0], 1), ([0, 0, 0], 2)])
def test_two_pass_matches_unsplit_loss_and_every_gradient(groups, microbatch):
    seed_training(7); m=model(); x=condition(len(groups))
    target=normalize_predictions(torch.randn(len(groups), 128, 2048)).detach().requires_grad_(True)
    ids=torch.tensor(groups)
    whole=jepa_prediction_loss(m(x).features, target, ids, **LOSS_RULE); whole.total.backward()
    gradients=[p.grad.clone() for p in m.parameters()]
    m.zero_grad(set_to_none=True)
    split=accumulate_loss(m, x, target, ids, microbatch=microbatch, loss_rule=LOSS_RULE)
    assert abs(split["loss"]-whole.total.item())<1e-6 and target.grad is None
    assert split["independent_anchors"]==len(set(groups))
    if len(set(groups))==1:assert split["variance"]==0
    else:assert split["variance"]>0
    errors=torch.cat([(p.grad-g).flatten() for p,g in zip(m.parameters(), gradients, strict=True)])
    reference=torch.cat([g.flatten() for g in gradients])
    assert (errors.norm()/reference.norm()).item()<1e-5
    assert errors.abs().max()<1e-5


def test_microbatch_penalty_averaging_is_detectably_wrong():
    seed_training(4); m=model(); x=condition(4)
    target=normalize_predictions(torch.randn(4,128,2048))
    pred=m(x).features
    correct=jepa_prediction_loss(pred,target,torch.tensor([0,0,1,1]),**LOSS_RULE)
    naive=sum(jepa_prediction_loss(pred[k:k+2],target[k:k+2],torch.zeros(2,dtype=torch.int64),**LOSS_RULE).total/2 for k in (0,2))
    assert correct.variance>0 and (correct.total-naive).item()>.01


def test_replay_fails_closed_for_stochastic_predictor():
    class RandomModel(torch.nn.Module):
        def forward(self,x):
            raw=torch.randn(len(x.features),128,2048,requires_grad=True)
            return SimpleNamespace(features=normalize_predictions(raw))
    with pytest.raises(ValueError,match="deterministic"):
        accumulate_loss(RandomModel(), condition(2), normalize_predictions(torch.randn(2,128,2048)),
                        torch.tensor([0,1]),microbatch=1,loss_rule=LOSS_RULE)


class TinyOrder:
    pair_counts=[3]*8
    @staticmethod
    def references(index):return [(0,index),(1,index)]


def engine():
    recipe={"optimizer":{"learning_rate":.0001,"betas":[.9,.999],"epsilon":1e-8,"weight_decay":.0001,
                          "warmup_updates":2,"minimum_lr_ratio":.1,"gradient_clip":1.},
            "total_updates":8,"predictor_microbatch":2,"loss":LOSS_RULE}
    return JepaTrainingEngine(model(),recipe,identity_digest="a"*64,indices=range(8),order=TinyOrder())


def step(e,**kwargs):
    x=condition(3); x=replace(x,state=x.state+random.random()+np.random.random())
    identities=[{"episode_index":ep,"anchor":e.global_step,"split":"train",
                 "pair_digest":f"{e.global_step*3+j:064x}"} for j,ep in enumerate((0,0,1))]
    target=normalize_predictions(torch.randn(3,128,2048))
    return e.step(x,target,identities,groups=anchor_groups(identities,target.device),**kwargs)


def test_jepa_disk_resume_restores_every_state_component_and_rng(tmp_path):
    seed_training(42); reference=engine()
    for _ in range(5):step(reference)
    expected=tree_digest(reference.snapshot())
    seed_training(42); interrupted=engine()
    for _ in range(2):step(interrupted)
    store=PixelCheckpointStore(tmp_path,"a"*64);store.save(interrupted.snapshot())
    seed_training(991);resumed=engine();resumed.restore(store.load())
    for _ in range(3):step(resumed)
    assert tree_digest(resumed.snapshot())==expected
    assert resumed.pairs_seen==15


def test_partial_two_pass_backward_cannot_commit(tmp_path):
    e=engine();step(e);store=PixelCheckpointStore(tmp_path,"a"*64);store.save(e.snapshot())
    def kill(start,stop):raise RuntimeError("killed after first replay chunk")
    with pytest.raises(RuntimeError):step(e,hook=kill)
    with pytest.raises(ValueError,match="partial"):e.snapshot()
    other=engine();other.restore(store.load());assert other.global_step==1


def test_wrong_anchor_groups_and_nontraining_samples_rejected():
    e=engine();x=condition(3);y=normalize_predictions(torch.randn(3,128,2048))
    identities=[{"episode_index":ep,"anchor":0,"split":"train","pair_digest":"0"*64} for ep in (0,0,1)]
    with pytest.raises(ValueError,match="groups"):
        e.step(x,y,identities,groups=torch.tensor([0,1,2]))
    identities[0]["split"]="validation"
    with pytest.raises(ValueError,match="split"):
        e.step(x,y,identities,groups=torch.tensor([0,0,1]))


def sample(ep,frame,split="train"):
    return SimpleNamespace(identity={"episode_index":ep,"target":frame,"anchor":0,"split":split},
                           target_rgb=np.full((2,256,256,3),frame,dtype=np.uint8))


class Teacher:
    def __init__(self):self.frames=0
    def __call__(self,rgb):
        self.frames+=len(rgb)
        return normalize_predictions(torch.stack([torch.randn(128,2048,generator=torch.Generator().manual_seed(int(x[0,0,0,0]))) for x in rgb]))


def test_target_cache_deduplication_cross_batch_hits_eviction_and_restart():
    teacher=Teacher();cache=PoolTargetCache(teacher)
    values,info=cache.encode([sample(0,2),sample(0,2),sample(1,4)],pool=0,device=torch.device('cpu'))
    assert teacher.frames==2 and info['target_cache_hits']==1
    hit,_=cache.encode([sample(0,2)],pool=0,device=torch.device('cpu'))
    assert teacher.frames==2 and torch.equal(hit[0],values[0])
    cache.encode([sample(2,8)],pool=1,device=torch.device('cpu'))
    assert list(cache.values)==[(2,8)] and teacher.frames==3
    cache.clear();fresh,_=cache.encode([sample(0,2)],pool=0,device=torch.device('cpu'))
    assert torch.equal(fresh[0],values[0]) and teacher.frames==4


@pytest.mark.parametrize('bad',['split','bytes','pool'])
def test_target_cache_rejects_invalid_inputs_before_encoding(bad):
    teacher=Teacher();cache=PoolTargetCache(teacher,maximum_bytes=TARGET_BYTES)
    samples=[sample(0,2)]
    if bad=='split':samples=[sample(0,2,'test')]
    elif bad=='bytes':samples.append(sample(0,4))
    else:samples=[sample(ep,2) for ep in range(5)]
    with pytest.raises(ValueError):cache.encode(samples,pool=0,device=torch.device('cpu'))
    assert teacher.frames==0
