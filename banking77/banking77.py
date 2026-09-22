"""Reproducible supervised adaptation of the original Laya decision architecture."""
import argparse, contextlib, functools, hashlib, json, math, os, random, time
from pathlib import Path
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import save_file, load_file
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import AutoConfig
from laya import Agent
from laya.common import build_sequence, collate_items, render_options

ROOT = Path(os.environ.get('LAYA_BANKING77_ROOT', Path(__file__).resolve().parent)).resolve()
def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(8)
def prepare():
    api = HfApi()
    model_revision = '1c5edc17a7acd8701df6fc341c0d179f1c62c982'
    data_revision = '18072d2685ea682290f7b8924d94c62acc19c0b2'
    if (ROOT/'model_download.json').exists():
        model=json.loads((ROOT/'model_download.json').read_text())['path']
    else:
        model = snapshot_download('convaiinnovations/laya', revision=model_revision,
            allow_patterns=['rl_agent_config.json','model.safetensors','tokenizer/*','encoder/*.json'], max_workers=4)
    # Laya weights already contain the encoder; only its architecture config is needed.
    cfg=json.loads((Path(model)/'rl_agent_config.json').read_text())
    enc=AutoConfig.from_pretrained(cfg['encoder']);enc.reference_compile=False
    enc.save_pretrained(Path(model)/'encoder')
    if all((ROOT/'raw_data'/f'{s}.jsonl').exists() for s in ['train','test']):
        ds=load_dataset('json',data_files={s:str(ROOT/'raw_data'/f'{s}.jsonl') for s in ['train','test']})
    else:
        ds = load_dataset('mteb/banking77', revision=data_revision)
    labels = sorted(set(ds['train']['label_text']))
    assert len(labels) == 77
    def rows(split):
        return [{'text': r['text'], 'label': labels.index(r['label_text'])} for r in ds[split]]
    train, test = rows('train'), rows('test')
    assert len(train) == 10003 and len(test) == 3080
    tr, va = train_test_split(np.arange(len(train)), test_size=.15, random_state=2026,
                             stratify=[r['label'] for r in train])
    data = {'train': [train[int(i)] for i in tr], 'validation': [train[int(i)] for i in va],
            'full_train': train, 'test': test, 'labels': labels}
    dump(ROOT/'data.json', data)
    dump(ROOT/'manifest.json', {'model': model, 'model_revision': model_revision,
         'dataset_revision': data_revision, 'train_indices': tr.tolist(), 'validation_indices': va.tolist(),
         'split_seed': 2026, 'dataset_sha256': hashlib.sha256((ROOT/'data.json').read_bytes()).hexdigest(),
         'upstream_commit':(ROOT/'UPSTREAM_COMMIT').read_text().strip(),
         'exact_train_test_text_overlap':len(set(r['text'] for r in train)&set(r['text'] for r in test)),
         'protocol': '77 classes; official 10003/3080 split; fixed stratified validation from training only; CE on original Laya decision logits; select by validation accuracy then macro F1; test only after selection.'})
    print('PREPARED', len(tr), len(va), len(test), flush=True)
def load_agent(path):
    a = Agent(str(path), device='cuda')
    assert a.device.type == 'cuda', 'GPU required; do not silently train on CPU'
    return a
def make_items(agent, rows, labels, original=False):
    q = {'t':'choice','ins':'Which banking intent does `message` express?',
         'crit':{x.replace('_',' '):None for x in labels}}
    if not original:
        option_ids = [[agent.tok.mask_token_id] + agent.tok(' '+s, add_special_tokens=False)['input_ids'] for s in render_options(q)]
        assert max(map(len, option_ids)) <= 49
        head = sum(map(len, option_ids)) + 64
        agent.cfg['head_max_len'] = head
        agent.cfg['max_len'] = int(math.ceil((head + 260)/64)*64)
    items=[]
    for r in rows:
        ids, markers = build_sequence(agent.tok, {'message':r['text']}, q,
            max_len=agent.cfg['max_len'], head_max_len=agent.cfg['head_max_len'])
        assert len(markers)==77
        items.append([{'ids':ids,'markers':markers,'qtype':0,'label':r['label']}])
    return items
def loader(agent, items, batch, shuffle=False):
    return DataLoader(items, batch_size=batch, shuffle=shuffle, num_workers=0,
        collate_fn=functools.partial(collate_items,pad_id=agent.tok.pad_token_id), pin_memory=True)
def forward(agent,b):
    keys=['input_ids','attention_mask','marker_pos','marker_mask','qtype']
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return agent.model(**{k:b[k].cuda(non_blocking=True) for k in keys})[0]
@torch.inference_mode()
def evaluate(agent, batches):
    agent.model.eval(); ys=[]; ps=[]
    for b in batches:
        logits=forward(agent,b)
        assert torch.isfinite(logits).all()
        ys.extend(b['label'].tolist());ps.extend(logits.argmax(-1).cpu().tolist())
    return {'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro'),
            'n':len(ys)},ys,ps
def save(agent, path):
    path.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in agent.model.state_dict().items()},str(path/'model.safetensors'))
    agent.model.encoder.config.save_pretrained(path/'encoder');agent.tok.save_pretrained(path/'tokenizer')
    cfg=dict(agent.cfg);cfg['temperature']=[1.,1.,1.];cfg['temperature_by_options']={}
    dump(path/'rl_agent_config.json',cfg)
def train(args):
    seed_all(args.seed)
    data=json.loads((ROOT/'data.json').read_text()); manifest=json.loads((ROOT/'manifest.json').read_text())
    out=ROOT/'runs'/args.name;out.mkdir(parents=True,exist_ok=True)
    dump(out/'config.json',vars(args))
    a=load_agent(manifest['model'])
    if args.gradient_checkpointing:
        a.model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    # The original action head is not trained by classification CE and remains intact.
    for p in a.model.act_head.parameters(): p.requires_grad_(False)
    items=make_items(a,data['full_train'] if args.full_train else data['train'],data['labels']); val=make_items(a,data['validation'],data['labels'])
    batches=loader(a,items,args.micro,True);vb=loader(a,val,args.micro)
    accum=args.batch//args.micro;assert args.batch%args.micro==0
    groups=[{'params':[p for n,p in a.model.named_parameters() if p.requires_grad and p.ndim>=2], 'weight_decay':args.wd},
            {'params':[p for n,p in a.model.named_parameters() if p.requires_grad and p.ndim<2], 'weight_decay':0.}]
    opt=torch.optim.AdamW(groups,lr=args.lr)
    steps=math.ceil(len(batches)/accum)*args.epochs
    sched=get_linear_schedule_with_warmup(opt,int(steps*args.warmup),steps)
    best=(-1.,-1.);best_epoch=0;start=time.time();history=[]
    for epoch in range(1,args.epochs+1):
        a.model.train();opt.zero_grad(set_to_none=True);loss_sum=0.
        for i,b in enumerate(batches):
            logits=forward(a,b);loss=torch.nn.functional.cross_entropy(logits,b['label'].cuda())
            assert torch.isfinite(loss), 'Nonfinite loss'
            # Correct normalization for the final partial accumulation group.
            group_size=min(accum,len(batches)-(i//accum)*accum)
            (loss/group_size).backward();loss_sum+=float(loss.detach())
            if (i+1)%accum==0 or i+1==len(batches):
                torch.nn.utils.clip_grad_norm_(a.model.parameters(),1.);opt.step();sched.step();opt.zero_grad(set_to_none=True)
            if (i+1)%100==0: print(json.dumps({'epoch':epoch,'batch':i+1,'batches':len(batches),'loss':loss_sum/(i+1),'seconds':time.time()-start}),flush=True)
        if args.full_train:
            metrics={'epoch':epoch,'train_loss':loss_sum/len(batches),'seconds':time.time()-start}
            history.append(metrics);dump(out/'history.json',history);print(json.dumps(metrics),flush=True)
            if epoch==args.epochs:
                save(a,out/'best');dump(out/'done.json',{'full_train':True,'epochs':epoch,'config':vars(args),'seconds':time.time()-start})
            continue
        metrics,_,_=evaluate(a,vb);metrics.update(epoch=epoch,train_loss=loss_sum/len(batches),seconds=time.time()-start)
        history.append(metrics);dump(out/'history.json',history);print(json.dumps(metrics),flush=True)
        score=(metrics['accuracy'],metrics['macro_f1'])
        if score>best:
            best=score;best_epoch=epoch;save(a,out/'best');dump(out/'best.json',metrics)
        elif epoch-best_epoch>=3: break
    if not args.full_train:dump(out/'done.json',{'best_accuracy':best[0],'best_macro_f1':best[1],'best_epoch':best_epoch,'seconds':time.time()-start,'config':vars(args)})
def final(args):
    seed_all(42);data=json.loads((ROOT/'data.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
    runs=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/done.json') if p.parent.name!='full_train_refit']
    assert runs
    best=max(runs,key=lambda r:(r['best_accuracy'],r['best_macro_f1']))
    dump(ROOT/'selection.json',best) # Written before the first test evaluation.
    while not (ROOT/'CLASSIFIER_READY').exists():
        print('Waiting for preregistered encoder-classifier comparison before test evaluation',flush=True)
        time.sleep(30)
    classifier_best=json.loads((ROOT/'classifier_selection_before_test.json').read_text())
    overall='native_laya' if (best['best_accuracy'],best['best_macro_f1']) >= (classifier_best['best_accuracy'],classifier_best['best_macro_f1']) else 'laya_encoder_classifier'
    dump(ROOT/'overall_selection_before_test.json',{'selected_architecture':overall,'native':best,'classifier':classifier_best,'criterion':'validation accuracy, then macro F1; locked before test'})
    a=load_agent(manifest['model']);results={}
    for name,original in [('laya_default',True),('laya_full_options',False)]:
        items=make_items(a,data['test'],data['labels'],original=original)
        metrics,ys,ps=evaluate(a,loader(a,items,8));results[name]=metrics
        dump(ROOT/'evaluation'/f'{name}_predictions.json',{'gold':ys,'predicted':ps})
    del a;torch.cuda.empty_cache()
    a=load_agent(ROOT/'runs'/'full_train_refit'/'best')
    items=make_items(a,data['test'],data['labels']);metrics,ys,ps=evaluate(a,loader(a,items,8))
    results['laya_finetuned']=metrics
    dump(ROOT/'evaluation'/'best_predictions.json',{'gold':ys,'predicted':ps})
    dump(ROOT/'evaluation'/'classification_report.json',classification_report(ys,ps,target_names=data['labels'],output_dict=True))
    dump(ROOT/'results.json',{'selected':best,'test':results,'experiments':runs})
    link=ROOT/'best_checkpoint'
    if not link.exists():link.symlink_to(ROOT/'runs'/'full_train_refit'/'best',target_is_directory=True)
    print(json.dumps(results,indent=2),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','final'])
    p.add_argument('--name',default='baseline');p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--micro',type=int,default=8)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--wd',type=float,default=.01)
    p.add_argument('--warmup',type=float,default=.06);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--full-train',action='store_true')
    p.add_argument('--gradient-checkpointing',action='store_true')
    args=p.parse_args()
    {'prepare':lambda:prepare(),'train':lambda:train(args),'final':lambda:final(args)}[args.mode]()
if __name__=='__main__':main()
