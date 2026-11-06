"""Open the prepared report capture only after the fixed pilot branch is decided."""
import argparse
import json
import shutil
from pathlib import Path
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--decision',type=Path,required=True);a=p.parse_args()
    b=Path('docs/evaluation/week10-weak-rlvr-v1');out=b/'event-report';decision=sealed(a.decision)
    protocol=sealed('configs/training/week10_weak_rlvr_evaluation_v1.json');plan=sealed(out/'locked-panel-plan.json')
    if (decision['protocol_digest']!=protocol['digest'] or decision['all_training_decisions_sealed'] is not True
            or decision['full_training_launched'] is not False or decision['calibration_permitted'] is not False):
        raise ValueError('Training decisions are not sealed; report remains locked')
    if plan['protocol_digest']!=protocol['digest'] or plan['panel']!='event-report':raise ValueError('Report plan mismatch')
    if (out/'panel-contract.json').exists():raise FileExistsError('Report already activated')
    future=sealed(b/'future-main-control-plan.json')
    if future['pilot_decision_digest']!=decision['digest'] or future['training_execution_authorized'] is not False:
        raise ValueError('Prospective full-run scope is not frozen')
    shutil.copy2(b/'future-main-control-plan.json',out/'future-main-control-plan.json')
    shutil.copy2(a.decision,out/'training-decision.json')
    c={k:v for k,v in plan.items() if k!='digest'}
    c.update(training_decision_digest=decision['digest'],report_payload_open_authorized=True,
        parent_locked_plan_digest=plan['digest'],future_main_control_plan_digest=future['digest'],report_model_inference_authorized=False,
        activation_authorization='User-approved steps1-4 include heldout preparation; predetermined pilot decision and resulting training scope are now sealed')
    c['bundle_sha256']={**c['bundle_sha256'],'training-decision.json':sha(out/'training-decision.json'),'future-main-control-plan.json':sha(out/'future-main-control-plan.json')}
    with (out/'panel-contract.json').open('x') as f:json.dump(seal_contract(c),f,indent=2,sort_keys=True);f.write('\n')
    print(sealed(out/'panel-contract.json')['digest'])
if __name__=='__main__':main()
