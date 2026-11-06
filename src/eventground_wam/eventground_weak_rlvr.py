"""Completion-bound exploratory SFT loading; original independent-QA gates are unchanged."""
from pathlib import Path
import torch
from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.pixel_predictor import tensor_state_digest


def load_completed_sft(model,allowlist,protocol,root):
    root=Path(root);completion=sealed(root/'run/completion.json');identity=sealed(root/'run/identity.json')
    if (completion['digest']!=protocol['sft_completion_digest'] or completion['status']!='complete'
            or completion['global_step']!=168 or completion['prompts_seen']!=2664 or completion['partial_update_in_progress']
            or completion['identity_digest']!=identity['digest'] or completion['endpoint_sha256']!=protocol['sft_endpoint_sha256']
            or not all(completion[k] for k in ('frozen_parameters_unchanged','stored_M4_parent_unchanged','source_files_unchanged','finite_updates'))):
        raise ValueError('SFT completion identity or fixed endpoint changed')
    path=root/'run'/completion['endpoint_file']
    if sha(path)!=protocol['sft_endpoint_sha256'] or path.with_suffix('.pt.sha256').read_text().strip()!=sha(path):raise ValueError('SFT endpoint checksum changed')
    payload=torch.load(path,map_location='cpu',weights_only=True);h=payload['header'];state=payload['parameters']
    names=sorted((*allowlist['head'],*allowlist['language_lora']));parameters=dict(model.named_parameters())
    if (h!=completion['inference_header'] or h['digest']!=contract_digest(h) or sorted(state)!=names or h['exact_names']!=names
            or h['parameter_sha256']!=protocol['sft_endpoint_tensor_digest'] or tensor_state_digest(state)!=h['parameter_sha256']
            or h['selection']!='fixed_epoch_2_update_168' or h['independent_qa_passed'] is not False):
        raise ValueError('SFT export tensor/header binding differs')
    with torch.no_grad():
        for name,tensor in state.items():
            p=parameters[name]
            if tensor.shape!=p.shape or tensor.dtype!=p.dtype or not torch.isfinite(tensor).all():raise ValueError('SFT tensor mismatch')
            p.copy_(tensor.to(p.device))
    if tensor_state_digest({n:parameters[n] for n in names})!=h['parameter_sha256']:raise ValueError('SFT live load mismatch')
    return h
