"""Independent checks of probability and source-paired report calculations."""
import importlib.util
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path('scripts').resolve()))
from analyze_week10_future_onset import calibration,baselines
from analyze_week10_weak_pilot import paired_bootstrap


def test_probability_calibration_perfect_and_confident_wrong():
    row=dict(target=dict(next_event='grasp'),lead_time_bin='1-4',event_probabilities=[0,0,0,1])
    perfect=calibration([row],'event_probabilities',['none','gripper_close','contact','grasp'],'event')
    assert perfect['multiclass_brier']==0 and perfect['ece_10_equal_width']==0
    row['event_probabilities']=[1,0,0,0]
    wrong=calibration([row],'event_probabilities',['none','gripper_close','contact','grasp'],'event')
    assert wrong['multiclass_brier']==2 and wrong['ece_10_equal_width']==1


def test_bootstrap_retains_all_prompts_within_source():
    rows=[]
    for source in ('a','b','c'):
        for kind in ('contact','grasp','gripper_close'):
            rows.append(dict(prompt_id=f'{source}-{kind}',source_id=source,target=dict(next_event=kind),positive_iou=.2))
    second=[dict(r,positive_iou=.4) for r in rows]
    result=paired_bootstrap(rows,second)
    assert result['sources']==3 and result['undefined_draws']==0
    assert result['interval95']==pytest.approx([.2,.2])
    with pytest.raises(ValueError,match='Unpaired'):paired_bootstrap(rows,list(reversed(second)))


def test_baseline_never_uses_report_labels_to_choose_predictions():
    def row(k,i):
        return dict(prompt_id=str(i),source_id='s',target=dict(next_event=k,interval=None if k=='none' else [2,5],horizon=None if k=='none' else 2),
            lead_time_bin='none' if k=='none' else '1-4',label_metadata=dict(end_censored=False))
    train=[row('none',0),row('none',1),row('grasp',2)]
    report=[row(k,i) for i,k in enumerate(('none','contact','grasp','gripper_close'))]
    b=baselines(train,report)
    assert b['training_only_prior']['prediction']['next_event']=='none'
    assert b['training_only_prior']['event_probabilities']==pytest.approx([2/3,0,0,1/3])
    assert b['always_none']['metrics']['macro_positive_iou']==0
