"""Explicit heterogeneous control: DeBERTa and Laya probability fusion."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from safetensors.torch import save_file,load_file
from transformers import AutoConfig,AutoModel,AutoTokenizer
from train_clinc import TextData,seed_all,infer,supcon,metrics,write_json
from round2 import development_data
from round3 import digest

ROOT=Path('artifacts/hybrid');BASE=ROOT/'base';LAYA='artifacts/round2/rdrop_42'

class HybridControl(nn.Module):
    def __init__(self,pretrained=True,base=BASE):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(str(base),local_files_only=True) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(str(base),local_files_only=True))
        self.dropout=nn.Dropout(.1);self.classifier=nn.Linear(self.encoder.config.hidden_size,150)
    def forward(self,ids,mask):
        h=self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state;w=mask[...,None].to(h.dtype)
        z=(h*w).sum(1)/w.sum(1).clamp_min(1)
        return self.classifier(self.dropout(z)).float(),F.normalize(z.float(),dim=-1)

def choose(prob,reference,y):
    rows=[{'laya_weight':a,**metrics((1-a)*prob+a*reference,y)} for a in [0.,.25,.5,.75,1.]]
    return max(rows,key=lambda r:(r['accuracy'],-r['nll'])),rows

def train(seed):
    assert (ROOT/'BASE_READY').exists();torch.set_num_threads(6);seed_all(seed)
    out=ROOT/f'deberta_{seed}';assert not out.exists();out.mkdir()
    data,labels=development_data();tok=AutoTokenizer.from_pretrained(BASE,use_fast=False,local_files_only=True)
    tr=DataLoader(TextData(data['train'],labels,tok,64),batch_size=16,shuffle=True,pin_memory=True)
    va=DataLoader(TextData(data['val'],labels,tok,64),batch_size=32)
    old=np.load(Path(LAYA)/'validation.npz');reference=torch.from_numpy(old['logits']).softmax(-1).numpy()
    model=HybridControl().cuda()
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},{'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=.01)
    total=math.ceil(len(tr)/4)*8;warmup=int(total*.1)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/warmup,max(0.,(total-s)/(total-warmup))))
    cfg={'model':'microsoft/deberta-v3-large','revision':'64a8c8eab3e352a784c658aef62be1662607476f','seed':seed,'epochs':8,'max_len':64,'batch_size':16,'accumulation':4,
         'encoder_lr':1e-5,'head_lr':1e-4,'supcon_weight':.1,'labels':labels,'parameters':sum(p.numel() for p in model.parameters()),
         'source_sha256':digest(__file__),'base_manifest_sha256':digest(ROOT/'download_manifest.json'),'data_sha256':digest('data/data_full.json')}
    write_json(out/'config.json',cfg);best=(-1.,-float('inf'));started=time.monotonic()
    for epoch in range(1,9):
        model.train();losses=[];opt.zero_grad(set_to_none=True)
        for step,(ids,mask,y) in enumerate(tr):
            y=y.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                l,z=model(ids.cuda(),mask.cuda());loss=F.cross_entropy(l,y)+.1*supcon(z,y)
            (loss/min(4,len(tr)-(step//4)*4)).backward();losses.append(float(loss.detach()))
            if (step+1)%4==0 or step+1==len(tr):
                nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();opt.zero_grad(set_to_none=True)
            if step%150==0:print(json.dumps({'epoch':epoch,'step':step,'loss':losses[-1],'seconds':time.monotonic()-started}),flush=True)
        l,_,y=infer(model,va);p=l.softmax(-1).numpy();y=y.numpy();assert np.array_equal(y,old['labels'])
        selection,grid=choose(p,reference,y);pure=metrics(p,y)
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),**selection,'pure_accuracy':pure['accuracy'],'seconds':time.monotonic()-started}
        with open(out/'history.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if (selection['accuracy'],-selection['nll'])>best:
            best=(selection['accuracy'],-selection['nll'])
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
            write_json(out/'selection.json',{'epoch':epoch,**selection,'test_used':False});write_json(out/'validation_grid.json',grid)
            np.savez_compressed(out/'validation.npz',probabilities=p,labels=y)
    (out/'COMPLETE').touch();print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);a=p.parse_args();train(a.seed)
