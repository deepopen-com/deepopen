"""Run a trained CLINC150 specialist on arbitrary text."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast
from train_clinc import IntentModel


def predict(checkpoint, texts, base='artifacts/base', device=None):
    folder=Path(checkpoint)
    if (folder/'base').is_dir():
        base=str(folder/'base')
    cfg=json.loads((folder/'config.json').read_text())
    mode=json.loads((folder/'inference_config.json').read_text()).get('variant') if (folder/'inference_config.json').exists() else None
    selected=json.loads((folder/'selection.json').read_text())
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok=PreTrainedTokenizerFast.from_pretrained(str(Path(base)/'tokenizer'))
    model=IntentModel(base)
    model.load_state_dict(load_file(str(folder/'model.safetensors')),strict=True)
    model.to(device).eval()
    tokens=tok(texts,padding='max_length',truncation=True,max_length=cfg['max_len'],return_tensors='pt')
    with torch.inference_mode():
        if device.startswith('cuda'):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,z=model(tokens['input_ids'].to(device),tokens['attention_mask'].to(device))
        else:
            logits,z=model(tokens['input_ids'].to(device),tokens['attention_mask'].to(device))
        if mode=='classifier_only':
            prob=logits.softmax(-1)
            return [{'text':text,'intent':cfg['labels'][int(p.argmax())],
                     'probability':float(p.max())} for text,p in zip(texts,prob)]
        prototypes=load_file(str(folder/'prototypes.safetensors'))['prototypes'].to(device)
        ce=(logits/selected['logit_temperature']).softmax(-1)
        pp=(z@prototypes.T/selected['prototype_temperature']).softmax(-1)
        prob=(1-selected['alpha'])*ce+selected['alpha']*pp
        if (folder/'retrieval_selection.json').exists():
            retrieval=json.loads((folder/'retrieval_selection.json').read_text())
            if retrieval['alpha']>0:
                memory=load_file(str(folder/'retrieval_memory.safetensors'))
                scores,idx=(z@memory['features'].to(device).T).topk(retrieval['k'],dim=-1)
                weights=(scores/retrieval['temperature']).softmax(-1)
                nearest=torch.zeros_like(prob)
                nearest.scatter_add_(1,memory['labels'].to(device)[idx],weights)
                nearest=nearest/nearest.sum(-1,keepdim=True).clamp_min(1e-12)
                prob=(1-retrieval['alpha'])*prob+retrieval['alpha']*nearest
            prob=(prob.clamp_min(1e-30).log()/retrieval['calibration_temperature']).softmax(-1)
        elif (folder/'calibration.json').exists():
            calibration=json.loads((folder/'calibration.json').read_text())
            prob=(prob.clamp_min(1e-30).log()/calibration['temperature']).softmax(-1)
    return [{'text':text,'intent':cfg['labels'][int(p.argmax())],
             'probability':float(p.max())} for text,p in zip(texts,prob)]


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--base',default='artifacts/base')
    p.add_argument('--text',action='append',required=True)
    a=p.parse_args()
    print(json.dumps(predict(a.checkpoint,a.text,a.base),ensure_ascii=False,indent=2))
