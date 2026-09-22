"""Inference for a transparently identified Laya/DeBERTa probability system."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,PreTrainedTokenizerFast
from safetensors.torch import load_file
from hybrid import HybridControl
from train_clinc import IntentModel,TextData,infer

def predict(checkpoint,texts):
    torch.set_num_threads(6);folder=Path(checkpoint);cfg=json.loads((folder/'system.json').read_text())
    labels=cfg['labels'];rows=[(text,labels[0]) for text in texts];a=cfg['laya_weight'];result=np.zeros((len(texts),150),dtype=np.float32)
    if a<1:
        tok=AutoTokenizer.from_pretrained(folder/'deberta_base',use_fast=False,local_files_only=True)
        loader=DataLoader(TextData(rows,labels,tok,64),batch_size=32)
        for name in cfg['deberta_members']:
            model=HybridControl(pretrained=False,base=folder/'deberta_base')
            model.load_state_dict(load_file(str(folder/name/'model.safetensors')),strict=True);model.cuda()
            logits,_,_=infer(model,loader);result+=(1-a)*logits.softmax(-1).numpy()/len(cfg['deberta_members'])
            del model;torch.cuda.empty_cache()
    if a>0:
        base=folder/'laya/base';tok=PreTrainedTokenizerFast.from_pretrained(base/'tokenizer')
        loader=DataLoader(TextData(rows,labels,tok,64),batch_size=64)
        model=IntentModel(str(base));model.load_state_dict(load_file(str(folder/'laya/model.safetensors')),strict=True);model.cuda()
        logits,_,_=infer(model,loader);result+=a*logits.softmax(-1).numpy()
        del model;torch.cuda.empty_cache()
    return [{'text':text,'intent':labels[int(p.argmax())],'probability':float(p.max()),'laya_weight':a,'deberta_members':len(cfg['deberta_members'])} for text,p in zip(texts,result)]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--text',action='append',required=True);a=p.parse_args()
    print(json.dumps(predict(a.checkpoint,a.text),indent=2,ensure_ascii=False))
