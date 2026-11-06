"""One prospective primary gate, at most one predefined calibration; no report tuning."""
import argparse
import json
from pathlib import Path
from analyze_week10_weak_pilot import read_eval,paired_bootstrap
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed
from eventground_wam.eventground_weak_evaluation import promotion
from eventground_wam.eventground_future_runtime import validate_training_completion


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--stage',choices=['primary','calibration'],required=True)
    a=parser.parse_args();base=a.base;p=sealed(base/'protocol.json')
    sft,first=read_eval(base/'eval/dev/sft');pilot,second=read_eval(base/'eval/dev'/a.stage)
    done=validate_training_completion(base,a.stage)
    if any(x['protocol_digest']!=p['digest'] for x in (sft,pilot,done)) or sft['manifest_digest']!=pilot['manifest_digest']:raise ValueError('Evaluation binding differs')
    if pilot['endpoint_completion_digest']!=done['digest']:raise ValueError('Wrong evaluated endpoint')
    if a.stage=='calibration' and not sealed(base/'decisions/primary.json')['calibration_permitted']:raise ValueError('No calibration admission')
    gate=promotion(sft['metrics'],pilot['metrics'],all(done['gates'].values()))
    d=seal_contract(dict(artifact_type='eventground_wam.future_onset_dev_decision',protocol_digest=p['digest'],
        stage=a.stage,promoted=gate['promoted'],gate=gate,selected_pilot=a.stage if gate['promoted'] else None,
        calibration_permitted=a.stage=='primary' and not gate['promoted'],
        sft_eval_digest=sft['digest'],pilot_eval_digest=pilot['digest'],attempt_digest=done['digest'],
        bootstrap=paired_bootstrap(first,second),independent_qa_passed=False,
        disposition='fresh SFT -> separate main and matched control' if gate['promoted'] else 'one predefined calibration' if a.stage=='primary' else 'stop RLVR main/control; preserve negative result'))
    (base/'decisions').mkdir(exist_ok=True)
    for name in [a.stage]+(['final'] if gate['promoted'] or a.stage=='calibration' else []):
        with (base/'decisions'/f'{name}.json').open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(d,indent=2))
if __name__=='__main__':main()
