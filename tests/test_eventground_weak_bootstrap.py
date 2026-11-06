import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('weak_analysis',Path(__file__).parents[1]/'scripts/analyze_week10_weak_pilot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_bootstrap_pairs_sources_and_constant_gain():
    rows=[dict(source_id=str(i),prompt_id=f'{i}-{k}',target=dict(next_event=k),positive_iou=.2) for i in range(3) for k in ('contact','grasp','gripper_close')]
    other=[{**r,'positive_iou':.3} for r in rows]
    x=m.paired_bootstrap(rows,other)
    assert x['undefined_draws']==0
    assert abs(x['interval95'][0]-.1)<1e-10 and abs(x['interval95'][1]-.1)<1e-10

def test_bootstrap_withholds_missing_class_draws():
    rows=[dict(source_id=str(i),prompt_id=str(i),target=dict(next_event=k),positive_iou=.2) for i,k in enumerate(('contact','grasp','gripper_close'))]
    x=m.paired_bootstrap(rows,rows)
    assert x['undefined_fraction']>.01 and x['interval95'] is None
