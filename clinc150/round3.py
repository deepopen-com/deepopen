"""Train-only distillation and late checkpoint averaging; validation selection only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from transformers import PreTrainedTokenizerFast
from train_clinc import IntentModel, TextData, infer, metrics, seed_all, supcon, write_json
from round2 import development_data, validation_loader, configure_dropout, symmetric_kl

ROOT=Path('artifacts/round3')
BASE='artifacts/base'
TEACHERS=[f'artifacts/round2/rdrop_{seed}' for seed in [13,42,87]]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load_model(path=None):
    model=IntentModel(BASE)
    if path: model.load_state_dict(load_file(str(Path(path)/'model.safetensors')),strict=True)
    else: model.load_laya_encoder(BASE)
    return model


def save_state(state,path):
    save_file({k:v.detach().cpu().contiguous() for k,v in state.items()},str(path))


def record_validation(model,loader,out):
    l,_,y=infer(model,loader)
    p=l.softmax(-1).numpy()
    result=metrics(p,y.numpy())
    np.savez_compressed(out/'validation.npz',logits=l.numpy(),labels=y.numpy())
    write_json(out/'validation_metrics.json',result)
    return result


def finish_config(out,cfg,result):
    write_json(out/'config.json',cfg)
    write_json(out/'selection.json',{'variant':'classifier_only','alpha':0.,'logit_temperature':1.,
                                    'prototype_temperature':.05,**result,'test_used_for_selection':False})
    write_json(out/'inference_config.json',{'variant':'classifier_only'})


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    target=ROOT/'teacher_train.npz'
    if target.exists(): raise RuntimeError('Teacher targets already exist')
    torch.set_num_threads(8)
    data,labels=development_data()
    tok=PreTrainedTokenizerFast.from_pretrained(f'{BASE}/tokenizer')
    loader=DataLoader(TextData(data['train'],labels,tok,64),batch_size=64,pin_memory=True)
    probability=None
    for path in TEACHERS:
        model=load_model(path).cuda()
        logits,_,y=infer(model,loader)
        p=(logits/2.).softmax(-1)
        probability=p/3 if probability is None else probability+p/3
        print('TEACHER_COMPLETE',path,flush=True)
        del model
        torch.cuda.empty_cache()
    probability=probability/probability.sum(-1,keepdim=True)
    np.savez_compressed(target,probabilities=probability.numpy(),labels=y.numpy())
    meta={'split':'train','temperature':2.,'members':TEACHERS,
          'teacher_sha256':{p:digest(Path(p)/'model.safetensors') for p in TEACHERS},
          'data_sha256':digest('data/data_full.json'),'targets_sha256':digest(target),
          'ordered_train_sha256':hashlib.sha256(json.dumps(data['train'],ensure_ascii=False).encode()).hexdigest(),
          'training_target_metrics':metrics(probability.numpy(),y.numpy()),
          'mean_entropy':float(-(probability*probability.clamp_min(1e-30).log()).sum(-1).mean())}
    write_json(ROOT/'teacher_metadata.json',meta)
    print(json.dumps(meta),flush=True)


def soup():
    torch.set_num_threads(8)
    out=ROOT/'soup_rdrop3'
    if out.exists(): raise RuntimeError('Soup already exists')
    out.mkdir(parents=True)
    average=None
    for path in TEACHERS:
        state=load_file(str(Path(path)/'model.safetensors'))
        if average is None: average={k:v.float()/3 for k,v in state.items()}
        else:
            for k,v in state.items(): average[k].add_(v.float(),alpha=1/3)
        del state
    model=load_model();model.load_state_dict(average);model.cuda()
    loader,labels=validation_loader()
    result=record_validation(model,loader,out)
    save_state(average,out/'model.safetensors')
    finish_config(out,{'method':'rdrop_weight_average','labels':labels,'max_len':64,'batch_size':64,
                      'members':TEACHERS,'parameters':sum(p.numel() for p in model.parameters())},result)
    print('SOUP_COMPLETE',json.dumps(result),flush=True)


class IndexedData(TextData):
    def __getitem__(self,i): return (*super().__getitem__(i),i)


def distillation_loss(logits,target,temperature=2.):
    return temperature**2*F.kl_div((logits/temperature).log_softmax(-1),target,reduction='batchmean')


def train(a):
    out=ROOT/f'{a.method}_{a.seed}'
    if out.exists(): raise RuntimeError(f'Refusing overwrite {out}')
    out.mkdir(parents=True)
    torch.set_num_threads(8); seed_all(a.seed)
    data,labels=development_data()
    tok=PreTrainedTokenizerFast.from_pretrained(f'{BASE}/tokenizer')
    ds=IndexedData(data['train'],labels,tok,64)
    loader=DataLoader(ds,batch_size=64,shuffle=True,pin_memory=True)
    val_loader,_=validation_loader()
    teacher=None
    if a.method=='distill':
        meta=json.loads((ROOT/'teacher_metadata.json').read_text())
        assert digest(ROOT/'teacher_train.npz')==meta['targets_sha256']
        assert digest('data/data_full.json')==meta['data_sha256']
        assert hashlib.sha256(json.dumps(data['train'],ensure_ascii=False).encode()).hexdigest()==meta['ordered_train_sha256']
        targets=np.load(ROOT/'teacher_train.npz')
        assert np.array_equal(targets['labels'],ds.y.numpy())
        teacher=torch.from_numpy(targets['probabilities'])
    epochs=8 if a.method=='distill' else 12
    lr=2e-5 if a.method=='distill' else 1e-5
    model=load_model();configure_dropout(model,.1);model.cuda()
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':lr},
                          {'params':model.classifier.parameters(),'lr':lr*10}],weight_decay=.01)
    total=epochs*len(loader);warmup=int(total*.1)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/warmup,max(0.,(total-s)/(total-warmup))))
    config={'method':a.method,'seed':a.seed,'epochs':epochs,'lr':lr,'head_lr':lr*10,
            'labels':labels,'max_len':64,'batch_size':64,'encoder_dropout':.1,'rdrop_weight':1.,
            'supcon_weight':.1,'distillation_weight':.5 if teacher is not None else 0.,'temperature':2.,
            'parameters':sum(p.numel() for p in model.parameters()),'late_average_epochs':list(range(epochs-3,epochs+1)),
            'source_sha256':digest(__file__),'teacher_targets_sha256':digest(ROOT/'teacher_train.npz') if teacher is not None else None}
    write_json(out/'config.json',config)
    best=(-1.,-float('inf')); average=None; count=0;start=time.monotonic()
    for epoch in range(epochs):
        model.train(); losses=[]
        for step,(ids,mask,y,idx) in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            ids,mask,y=ids.cuda(),mask.cuda(),y.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                l1,z1=model(ids,mask);l2,z2=model(ids,mask)
                ce=.5*(F.cross_entropy(l1,y)+F.cross_entropy(l2,y))
                con=.5*(supcon(z1,y)+supcon(z2,y))
                kl=symmetric_kl(l1,l2)
                kd=l1.sum()*0
                if teacher is not None:
                    target=teacher[idx].cuda()
                    kd=.5*(distillation_loss(l1,target)+distillation_loss(l2,target))
                loss=ce+.1*con+kl+.5*kd
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();scheduler.step()
            losses.append([float(x.detach()) for x in [ce,con,kl,kd]])
            if step%50==0:
                print(json.dumps({'epoch':epoch+1,'step':step,'ce_con_kl_kd':losses[-1],
                                  'seconds':time.monotonic()-start}),flush=True)
        l,_,y=infer(model,val_loader)
        result=metrics(l.softmax(-1).numpy(),y.numpy())
        row={'epoch':epoch+1,**result,'loss_components':np.mean(losses,axis=0).tolist(),
             'seconds':time.monotonic()-start}
        with open(out/'history.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if (result['accuracy'],-result['nll'])>best:
            best=(result['accuracy'],-result['nll'])
            save_state(model.state_dict(),out/'best_epoch.safetensors')
            write_json(out/'best_epoch.json',row)
        if epoch>=epochs-4:
            count+=1
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if average is None: average=state
            else:
                for k,v in state.items(): average[k].lerp_(v,1/count)
            del state
    variants={}
    for name,state in [('late_average',average),('best_epoch',load_file(str(out/'best_epoch.safetensors')))]:
        folder=out/name;folder.mkdir()
        model.load_state_dict(state)
        result=record_validation(model,val_loader,folder)
        save_state(state,folder/'model.safetensors')
        finish_config(folder,{**config,'checkpoint_variant':name},result)
        variants[name]=result
    choice=max(variants,key=lambda k:(variants[k]['accuracy'],-variants[k]['nll']))
    import shutil
    for name in ['model.safetensors','validation.npz','validation_metrics.json','selection.json','inference_config.json','config.json']:
        shutil.copy2(out/choice/name,out/name)
    write_json(out/'checkpoint_choice.json',{'chosen':choice,'variants':variants,'test_used':False})
    print('TRAIN_COMPLETE',out,choice,json.dumps(variants[choice]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','soup','train'])
    p.add_argument('--method',choices=['distill','long_rdrop']);p.add_argument('--seed',type=int,default=13)
    a=p.parse_args()
    {'prepare':prepare,'soup':soup,'train':lambda:train(a)}[a.action]()
