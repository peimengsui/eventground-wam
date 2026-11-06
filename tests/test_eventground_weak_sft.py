import copy
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from prepare_week10_weak_sft import BUNDLE, make_launch
from test_eventground_training import batches as tiny_batches
from test_eventground_training import setup as tiny_setup
from train_week10_weak_sft import save_checkpoint, save_payload

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_training import EventGroundTrainer, require_training_admission
from eventground_wam.eventground_weak_sft import (
    admission,
    cursor,
    load_inputs,
    schedule_contract,
    schedule_rows,
    validate_launch,
)
from eventground_wam.eventground_weak_supervision import sealed


def reseal(value):
    return seal_contract({k:v for k,v in value.items() if k!="digest"})


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root=tmp_path_factory.mktemp("weak-sft")/"root"
    shutil.copytree(ROOT/"docs/evaluation/week10-exploratory-weak-v4/source",root)
    for name in ["src/eventground_wam/eventground_weak_sft.py","scripts/prepare_week10_weak_sft.py",
                 "scripts/train_week10_weak_sft.py","docs/evaluation/week10-exploratory-weak-v4/run/completion.json"]:
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    dataset=sealed(root/"docs/evaluation/week10-exploratory-weak-v4/dataset.json")
    schedule=schedule_contract(dataset)
    path=root/BUNDLE/"schedule.json";path.parent.mkdir(parents=True);path.write_text(json.dumps(schedule))
    launch=make_launch(root,schedule)
    return root,launch,load_inputs(root,launch)


def test_explicit_new_admission_keeps_old_independent_gate_closed(prepared):
    root,launch,data=prepared
    batches=validate_launch(root,launch)
    assert len(batches)==168 and sum(map(len,batches))==2664
    admitted=admission(launch)
    assert admitted["formal_sft_authorized"] and not admitted["independent_qa_passed"]
    assert not admitted["rlvr_authorized"]
    with pytest.raises(ValueError,match="Formal training"):
        require_training_admission(admitted,study_digest=data["protocol"]["parent_study_digest"],
                                   recipe_digest=data["recipe"]["digest"],manifest_digest=data["dataset"]["digest"])


def test_two_passes_cover_each_prompt_once_with_unpadded_partial_batches(prepared):
    _,_,data=prepared
    dataset=data["dataset"];batches=schedule_rows(dataset)
    assert [len(batches[i]) for i in (83,167)]==[4,4]
    assert all(len(b)==16 for i,b in enumerate(batches) if i not in (83,167))
    ids={r["prompt_id"] for r in dataset["main_rows"]}
    for epoch in (0,1):
        seen=Counter(r["prompt_id"] for b in batches[epoch*84:(epoch+1)*84] for r in b)
        assert set(seen)==ids and set(seen.values())=={1}
    reverse=copy.deepcopy(dataset);reverse["main_rows"].reverse()
    assert schedule_rows(reseal(reverse))==batches
    assert batches[0]!=batches[84]


@pytest.mark.parametrize("step,prompts,epoch,batch",[(0,0,0,0),(83,1328,0,83),(84,1332,1,0),
    (167,2660,1,83),(168,2664,None,None)])
def test_cursor_handles_84_update_epochs(prepared,step,prompts,epoch,batch):
    schedule=prepared[2]["schedule"]
    value=cursor(step,prompts,schedule)
    assert value["next_epoch_index"]==epoch and value["next_batch_index"]==batch
    with pytest.raises(ValueError):cursor(step,prompts+1,schedule)


@pytest.mark.parametrize("field,value",[("epochs",3),("stage","rlvr"),("pilot_weights_reused",True),
    ("optimizer_state_reused",True),("independent_qa_passed",True),("formal_sft_authorized",False),
    ("maximum_attempts",2),("logical_updates",169),("heldout_payload_reads",1)])
def test_launch_scope_cannot_silently_expand(prepared,field,value):
    root,launch,_=prepared
    changed=copy.deepcopy(launch);changed[field]=value
    with pytest.raises(ValueError):validate_launch(root,reseal(changed))


@pytest.mark.parametrize("mutation",["failed_pilot","empty_gates","bad_recovery","resource","schedule"])
def test_evidence_and_resource_admission_fail_closed(prepared,mutation):
    root,launch,data=prepared
    launch=copy.deepcopy(launch);data=copy.deepcopy(data)
    if mutation=="resource":
        launch["resources"]["wall_seconds"]+=1
    elif mutation=="schedule":
        data["schedule"]["batches"][0].reverse();data["schedule"]=reseal(data["schedule"])
    else:
        q=data["qualification"]
        if mutation=="failed_pilot":q["status"]="failed"
        if mutation=="empty_gates":q["gates"]={}
        if mutation=="bad_recovery":q["recovery"]["actual_digest"]="0"*64
        data["qualification"]=reseal(q)
    with pytest.raises(ValueError):validate_launch(root,reseal(launch),data)


def test_new_checkpoint_is_exactly_recoverable_and_write_bounded(tmp_path,prepared):
    original=tiny_setup();recipe=copy.deepcopy(original.recipe);recipe["logical_batch_size"]=16
    original=EventGroundTrainer(original.model,original.allowlist,reseal(recipe))
    micro=tiny_batches()*3+[tiny_batches((1,4))[0]]
    original.sft_update(micro,prompt_count=16)
    identity=seal_contract({"stage":"weak-sft","launch":"test-only"})
    resources={"output_bytes":20*1024**2,"minimum_free_bytes":1}
    checkpoint=tmp_path/"step1.pt"
    save_checkpoint(original,checkpoint,identity,prepared[2]["schedule"],tmp_path,resources)
    marker=sealed(checkpoint.with_suffix(".pt.cursor.json"))
    assert marker["next_epoch_index"]==0 and marker["next_batch_index"]==1
    expected=original.sft_update(micro,prompt_count=16)
    params={n:p.detach().clone() for n,p in original.model.named_parameters()}
    original.resume(checkpoint,identity=identity)
    assert original.sft_update(micro,prompt_count=16)==expected
    assert all(torch.equal(params[n],p) for n,p in original.model.named_parameters())
    with pytest.raises((ValueError,FileExistsError)):
        save_payload(checkpoint,{},output=tmp_path,resources=resources)
    with pytest.raises(RuntimeError):
        save_payload(tmp_path/"too-big.pt",{"t":torch.ones(100)},output=tmp_path,
                     resources={"output_bytes":32,"minimum_free_bytes":1})
    assert not (tmp_path/"too-big.pt").exists()
