"""Explicit alternative: Laya encoder with a NEW fixed 77-way HF classifier head.
This is not the original Laya decision head and must be reported separately.
"""
import os
os.environ['USE_TF']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,f1_score,classification_report
from banking77 import ROOT,seed_all,dump
p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','test']);p.add_argument('--name',default='encoder_lr2e5');p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--epochs',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--full-train',action='store_true');p.add_argument('--checkpoint');args=p.parse_args()
seed_all(args.seed);data=json.loads((ROOT/'data.json').read_text());base=json.loads((ROOT/'manifest.json').read_text())['model'];out=ROOT/'classifier_runs'/args.name;out.mkdir(parents=True,exist_ok=True)
if args.mode=='test':
    model=AutoModelForSequenceClassification.from_pretrained(args.checkpoint,attn_implementation='sdpa');tok=AutoTokenizer.from_pretrained(args.checkpoint)
else:
    cfg=AutoConfig.from_pretrained(Path(base)/'encoder');cfg.num_labels=77;cfg.reference_compile=False
    cfg.id2label={i:x for i,x in enumerate(data['labels'])};cfg.label2id={x:i for i,x in enumerate(data['labels'])};cfg.classifier_dropout=.1
    model=AutoModelForSequenceClassification.from_config(cfg,attn_implementation='sdpa')
    state=load_file(str(Path(base)/'model.safetensors'))
    encoder={k[len('encoder.'):]:v for k,v in state.items() if k.startswith('encoder.')}
    model.model.load_state_dict(encoder,strict=True);del state,encoder
    tok=AutoTokenizer.from_pretrained(Path(base)/'tokenizer')
model.cuda()
def batches(split,shuffle=False):
    rows=data[split];enc=tok([r['text'] for r in rows],truncation=True,max_length=128)
    items=[{**{k:v[i] for k,v in enc.items()},'labels':r['label']} for i,r in enumerate(rows)]
    return DataLoader(items,batch_size=32,shuffle=shuffle,collate_fn=DataCollatorWithPadding(tok),pin_memory=True)
@torch.inference_mode()
def evaluate(dl):
    model.eval();ys=[];ps=[]
    for batch in dl:
        ys.extend(batch.pop('labels').tolist())
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**{k:v.cuda() for k,v in batch.items()}).logits
        assert torch.isfinite(logits).all();ps.extend(logits.argmax(-1).cpu().tolist())
    return {'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro'),'n':len(ys)},ys,ps
if args.mode=='test':
    metrics,y,pred=evaluate(batches('test'));dump(ROOT/'evaluation/classifier_predictions.json',{'gold':y,'predicted':pred,'metrics':metrics,'checkpoint':args.checkpoint});print(metrics,flush=True)
else:
    dump(out/'config.json',{**vars(args),'architecture':'Laya encoder + new Transformers ModernBERT 77-class head','head_lr_multiplier':10,'weight_decay':.01,'warmup_ratio':.1,'batch':32,'max_length':128,'parameter_count':sum(p.numel() for p in model.parameters())})
    train=batches('full_train' if args.full_train else 'train',True);val=batches('validation')
    groups=[]
    for is_encoder in [True,False]:
        for decay in [True,False]:
            params=[v for n,v in model.named_parameters() if n.startswith('model.')==is_encoder and (v.ndim>=2)==decay]
            groups.append({'params':params,'lr':args.lr*(1 if is_encoder else 10),'weight_decay':.01 if decay else 0.})
    optimizer=torch.optim.AdamW(groups);steps=len(train)*args.epochs;sched=get_linear_schedule_with_warmup(optimizer,int(steps*.1),steps)
    best=(-1.,-1.);best_epoch=0;history=[];start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train();total=0.
        for b in train:
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**{k:v.cuda(non_blocking=True) for k,v in b.items()}).loss
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();sched.step();optimizer.zero_grad(set_to_none=True);total+=float(loss.detach())
        if args.full_train:metrics={'epoch':epoch,'train_loss':total/len(train),'seconds':time.time()-start}
        else:
            metrics,_,_=evaluate(val);metrics.update(epoch=epoch,train_loss=total/len(train),seconds=time.time()-start)
        history.append(metrics);dump(out/'history.json',history);print(json.dumps(metrics),flush=True)
        score=(metrics.get('accuracy',-1),metrics.get('macro_f1',-1))
        if (args.full_train and epoch==args.epochs) or (not args.full_train and score>best):
            if not args.full_train:best=score;best_epoch=epoch
            model.save_pretrained(out/'best');tok.save_pretrained(out/'best');dump(out/'best.json',metrics)
        if not args.full_train and epoch-best_epoch>=3:break
    dump(out/'done.json',{'best_accuracy':best[0],'best_macro_f1':best[1],'best_epoch':best_epoch,'config':vars(args),'seconds':time.time()-start})
