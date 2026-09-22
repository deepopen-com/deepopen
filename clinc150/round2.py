"""Round-two validation-only development; test scoring is a separate frozen step."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from safetensors.torch import load_file, save_file
from transformers import PreTrainedTokenizerFast
from train_clinc import IntentModel, TextData, infer, metrics, seed_all, supcon, write_json

ROOT = Path('artifacts/round2')
BASE = 'artifacts/base'
OLD = [f'artifacts/runs/{m}_{s}' for m in ['ce', 'supcon'] for s in [13,42,87]]


def development_data():
    raw = json.loads(Path('data/data_full.json').read_text())
    # Deliberately discard the test split before any diagnostics or training.
    data = {s: raw[s] for s in ['train', 'val']}
    labels = sorted({y for _, y in data['train']})
    assert len(labels) == 150 and len(data['train']) == 15000 and len(data['val']) == 3000
    return data, labels


def validation_loader(batch=64):
    data, labels = development_data()
    tok = PreTrainedTokenizerFast.from_pretrained(f'{BASE}/tokenizer')
    ds = TextData(data['val'], labels, tok, 64)
    return DataLoader(ds, batch_size=batch, pin_memory=True), labels


def score_and_save(model, loader, out):
    logits, z, y = infer(model, loader)
    score = metrics(logits.softmax(-1).numpy(), y.numpy())
    np.savez_compressed(out/'validation.npz', logits=logits.numpy(), labels=y.numpy())
    write_json(out/'validation_metrics.json', score)
    return score


def ensemble_groups():
    return {'ce3': OLD[:3], 'supcon3': OLD[3:], 'all6': OLD}


def diagnose():
    ROOT.mkdir(parents=True, exist_ok=True)
    data, labels = development_data()
    probs = {}
    errors = {}
    rows = []
    for path in OLD:
        v = np.load(Path(path)/'validation.npz')
        logits = torch.from_numpy(v['logits'])
        y = v['labels']
        p = logits.softmax(-1).numpy()
        probs[path] = p
        wrong = p.argmax(1) != y
        errors[path] = wrong
        rows.append({'path': path, **metrics(p,y)})
    groups = []
    for name, paths in ensemble_groups().items():
        p = np.mean([probs[x] for x in paths], axis=0)
        groups.append({'name': name, 'members': paths, **metrics(p,y)})
    common = np.stack(list(errors.values())).all(0)
    supwrong = np.stack([errors[x] for x in OLD[3:]])
    confusion = Counter((labels[int(y[i])], labels[int(probs[OLD[-1]][i].argmax())])
                        for i in np.flatnonzero(errors[OLD[-1]]))
    result = {'split': 'val', 'individual': rows, 'ensembles': groups,
              'all_six_wrong': int(common.sum()), 'supcon_all_wrong': int(supwrong.all(0).sum()),
              'supcon_any_wrong': int(supwrong.any(0).sum()),
              'seed87_confusions': [{'gold':a,'predicted':b,'count':n} for (a,b),n in confusion.most_common(20)],
              'batch64_expected_positive_anchor_fraction': 1-(1-1/150)**63}
    write_json(ROOT/'diagnostics.json', result)
    with open(ROOT/'validation_errors.jsonl','w',encoding='utf-8') as f:
        for i in np.flatnonzero(supwrong.any(0)):
            f.write(json.dumps({'index':int(i),'text':data['val'][i][0], 'gold':labels[int(y[i])],
                   'supcon_predictions':[labels[int(probs[x][i].argmax())] for x in OLD[3:]]})+'\n')
    print(json.dumps(result), flush=True)


class BalancedBatches(Sampler):
    """Exactly one visit/example/epoch: 150 classes x 25 blocks of 4 examples."""
    def __init__(self, y, seed):
        self.y = np.asarray(y)
        self.seed = seed
        self.epoch = 0
    def __len__(self):
        return 235  # ceil(15000 / 64), final batch has 24 examples
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        pools = {k: list(rng.permutation(np.flatnonzero(self.y == k)).reshape(-1,4)) for k in range(150)}
        blocks=[]
        for _ in range(25):
            for k in rng.permutation(150):
                blocks.append(pools[k].pop().tolist())
        for start in range(0,len(blocks),16):
            yield [i for block in blocks[start:start+16] for i in block]


def symmetric_kl(a,b):
    la,lb = a.log_softmax(-1), b.log_softmax(-1)
    return .5 * (F.kl_div(la, lb.exp(), reduction='batchmean') +
                 F.kl_div(lb, la.exp(), reduction='batchmean'))


def configure_dropout(model, value):
    # ModernBERT uses config.attention_dropout in SDPA and nn.Dropout in its MLPs.
    model.encoder.config.attention_dropout = value
    model.encoder.config.mlp_dropout = value
    model.encoder.config.embedding_dropout = value
    for module in model.encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = value
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = value
            if hasattr(module, 'out_drop'):
                module.out_drop = nn.Dropout(value)


def train(a):
    out=ROOT/f'{a.method}_{a.seed}'
    if out.exists():
        raise RuntimeError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    seed_all(a.seed)
    torch.set_num_threads(8)
    data, labels = development_data()
    tok = PreTrainedTokenizerFast.from_pretrained(f'{BASE}/tokenizer')
    ds = TextData(data['train'],labels,tok,64)
    if a.method == 'balanced_supcon':
        loader = DataLoader(ds,batch_sampler=BalancedBatches(ds.y.numpy(),a.seed),pin_memory=True)
    else:
        loader = DataLoader(ds,batch_size=64,shuffle=True,pin_memory=True)
    val_loader,_ = validation_loader()
    model=IntentModel(BASE)
    model.load_laya_encoder(BASE)
    if a.method == 'rdrop':
        configure_dropout(model, .1)
    model.cuda()
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':2e-5},
                           {'params':model.classifier.parameters(),'lr':2e-4}],weight_decay=.01)
    total=8*len(loader); warmup=max(1,int(total*.1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s+1)/warmup,max(0.,(total-s)/max(1,total-warmup))))
    config={'method':a.method,'seed':a.seed,'epochs':8,'lr':2e-5,'head_lr':2e-4,'batch_size':64,
            'max_len':64,'labels':labels,'contrastive_weight':.1,'rdrop_weight':1. if a.method=='rdrop' else 0.,
            'encoder_dropout':.1 if a.method=='rdrop' else 0.,'base':BASE,
            'parameters':sum(p.numel() for p in model.parameters()),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'test_used_for_selection':False}
    write_json(out/'config.json',config)
    best=(-1.,-float('inf')); start=time.monotonic()
    for epoch in range(8):
        model.train(); running=[]; positive=[]; steps=0
        for step,(ids,mask,y) in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            ids,mask,y=ids.cuda(),mask.cuda(),y.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                l,z=model(ids,mask)
                ce=F.cross_entropy(l,y); con=supcon(z,y)
                kl=l.sum()*0
                if a.method=='rdrop':
                    l2,z2=model(ids,mask)
                    ce=.5*(ce+F.cross_entropy(l2,y))
                    con=.5*(con+supcon(z2,y))
                    kl=symmetric_kl(l,l2)
                loss=ce+.1*con+kl
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); scheduler.step()
            running.append([float(ce.detach()),float(con.detach()),float(kl.detach())])
            positive.append(float((y[:,None].eq(y[None,:]).sum(1)>1).float().mean()))
            steps+=1
            if step%50==0:
                print(json.dumps({'epoch':epoch+1,'step':step,'ce_con_kl':running[-1],
                                  'seconds':time.monotonic()-start}),flush=True)
        vl,_,vy=infer(model,val_loader)
        m=metrics(vl.softmax(-1).numpy(),vy.numpy())
        row={'epoch':epoch+1,**m,'loss_components':np.mean(running,axis=0).tolist(),
             'positive_anchor_fraction':float(np.mean(positive)), 'steps':steps,
             'seconds':time.monotonic()-start}
        with open(out/'history.jsonl','a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if (m['accuracy'],-m['nll'])>best:
            best=(m['accuracy'],-m['nll'])
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
            write_json(out/'best_epoch.json',row)
    model.load_state_dict(load_file(str(out/'model.safetensors')))
    result=score_and_save(model,val_loader,out)
    write_json(out/'selection.json',{'variant':'classifier_only','alpha':0.,'logit_temperature':1.,
                                  'prototype_temperature':.05,**result,'test_used_for_selection':False})
    write_json(out/'inference_config.json',{'variant':'classifier_only'})
    print('COMPLETE',out,json.dumps(result),flush=True)


def soups():
    torch.set_num_threads(8)
    loader,labels=validation_loader()
    groups={f'paired_{s}':[f'artifacts/runs/{m}_{s}' for m in ['ce','supcon']] for s in [13,42,87]}
    groups.update(supcon3=OLD[3:], all6=OLD)
    for name,paths in groups.items():
        out=ROOT/f'soup_{name}'
        if (out/'validation_metrics.json').exists(): continue
        out.mkdir(parents=True,exist_ok=True)
        weights=None
        for path in paths:
            state=load_file(str(Path(path)/'model.safetensors'))
            if weights is None:
                weights={k:v.float()/len(paths) if v.is_floating_point() else v.clone() for k,v in state.items()}
            else:
                for k,v in state.items():
                    if v.is_floating_point(): weights[k].add_(v.float(),alpha=1/len(paths))
                    else: assert torch.equal(weights[k],v)
            del state
        model=IntentModel(BASE); model.load_state_dict(weights,strict=True); model.cuda()
        result=score_and_save(model,loader,out)
        save_file(weights,str(out/'model.safetensors'))
        write_json(out/'config.json',{'method':'weight_average','members':paths,'labels':labels,
                                   'max_len':64,'batch_size':64,'parameters':sum(p.numel() for p in model.parameters())})
        write_json(out/'selection.json',{'variant':'classifier_only','alpha':0.,'logit_temperature':1.,
                    'prototype_temperature':.05,**result,'test_used_for_selection':False})
        write_json(out/'inference_config.json',{'variant':'classifier_only'})
        print(name,json.dumps(result),flush=True)
        del weights,model
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['diagnose','soups','train'])
    p.add_argument('--method',choices=['balanced_supcon','rdrop'])
    p.add_argument('--seed',type=int,default=13)
    a=p.parse_args()
    if a.action=='diagnose': diagnose()
    elif a.action=='soups': soups()
    else: train(a)
