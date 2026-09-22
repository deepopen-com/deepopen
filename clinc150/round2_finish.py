"""Freeze validation decisions, then evaluate and package the selected round-two models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from train_clinc import IntentModel, TextData, infer, metrics, write_json
from round2 import ROOT, BASE, OLD, ensemble_groups


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def freeze():
    target=ROOT/'evaluation_freeze.json'
    if target.exists(): raise RuntimeError('Already frozen')
    assert (ROOT/'SCREENING_COMPLETE').exists()
    assert (ROOT/'REPLICATES_COMPLETE').exists()
    rows=[]
    for path in ROOT.iterdir():
        if path.is_dir() and (path/'validation_metrics.json').exists():
            m=json.loads((path/'validation_metrics.json').read_text())
            rows.append({'path':str(path),**m})
    selected=max(rows,key=lambda r:(r['accuracy'],-r['nll']))
    diagnostics=json.loads((ROOT/'diagnostics.json').read_text())
    # A completed replicated family must have all three seeds, else no multi-seed claim.
    families={}
    for method in ['balanced_supcon','rdrop']:
        paths=[ROOT/f'{method}_{s}' for s in [13,42,87]]
        if all((p/'selection.json').exists() for p in paths): families[method]=[str(p) for p in paths]
    ensemble_rows=list(diagnostics['ensembles'])
    for method,paths in families.items():
        raw=[np.load(Path(p)/'validation.npz') for p in paths]
        y=raw[0]['labels']
        assert all(np.array_equal(r['labels'],y) for r in raw)
        prob=np.mean([torch.from_numpy(r['logits']).softmax(-1).numpy() for r in raw],axis=0)
        ensemble_rows.append({'name':method+'3','members':paths,**metrics(prob,y)})
    ensemble=max(ensemble_rows,key=lambda r:(r['accuracy'],-r['nll']))
    test_paths=sorted(set([selected['path']]+[x for paths in families.values() for x in paths]))
    result={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'selection_rule':'validation accuracy descending, NLL ascending',
            'test_previously_observed':'Round-one aggregate scores known; no test-error-driven tuning',
            'all_validation_candidates':rows,'selected_single':selected,'selected_ensemble':ensemble,
            'all_ensemble_validation_candidates':ensemble_rows,
            'replicated_families':families,'test_paths':test_paths,
            'weight_sha256':{p:digest(Path(p)/'model.safetensors') for p in test_paths},
            'data_sha256':digest('data/data_full.json'),
            'source_sha256':{p:digest(p) for p in ['round2.py','round2_finish.py','ROUND2_PLAN.md']}}
    write_json(target,result)
    print(json.dumps(result),flush=True)


def paired(new,old,y):
    newok=new.argmax(1)==y; oldok=old.argmax(1)==y
    delta=newok.astype(float)-oldok.astype(float)
    rng=np.random.default_rng(20260921)
    boot=np.array([delta[rng.integers(0,len(y),len(y))].mean() for _ in range(10000)])
    gained=int((newok&~oldok).sum()); lost=int((oldok&~newok).sum())
    from scipy.stats import binomtest
    return {'accuracy_delta':float(delta.mean()),'gained':gained,'lost':lost,
            'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
            'mcnemar_exact_p':float(binomtest(gained,gained+lost,.5).pvalue) if gained+lost else 1.}


def evaluate():
    frozen=json.loads((ROOT/'evaluation_freeze.json').read_text())
    if (ROOT/'test_summary.json').exists(): raise RuntimeError('Test already evaluated')
    assert digest('data/data_full.json')==frozen['data_sha256']
    for name,sha in frozen['source_sha256'].items(): assert digest(name)==sha
    torch.set_num_threads(8)
    raw=json.loads(Path('data/data_full.json').read_text())
    labels=sorted({y for _,y in raw['train']})
    tok=PreTrainedTokenizerFast.from_pretrained(f'{BASE}/tokenizer')
    ds=TextData(raw['test'],labels,tok,64)
    loader=DataLoader(ds,batch_size=64,pin_memory=True)
    reference=np.load('artifacts/runs/supcon_87/test_predictions.npz')
    old=reference['classifier_probabilities']; gold=reference['labels']
    train_texts={t.strip().lower() for t,_ in raw['train']}
    clean=np.array([text.strip().lower() not in train_texts for text,_ in raw['test']])
    results={}
    for path in frozen['test_paths']:
        out=Path(path)
        assert digest(out/'model.safetensors')==frozen['weight_sha256'][path]
        if (out/'test_metrics.json').exists(): raise RuntimeError(f'Test already exists {out}')
        model=IntentModel(BASE)
        model.load_state_dict(load_file(str(out/'model.safetensors')),strict=True)
        model.cuda()
        l,_,y=infer(model,loader)
        y=y.numpy(); p=l.softmax(-1).numpy()
        assert np.array_equal(y,gold)
        m={'classifier_only':metrics(p,y),'excluding_train_duplicates':metrics(p[clean],y[clean]),
           'paired_vs_round1_supcon87':paired(p,old,y)}
        write_json(out/'test_metrics.json',m)
        np.savez_compressed(out/'test_predictions.npz',classifier_probabilities=p,labels=y)
        results[path]=m
        print(path,json.dumps(m),flush=True)
        del model
        torch.cuda.empty_cache()
    ens=frozen['selected_ensemble']
    members=[np.load(Path(p)/'test_predictions.npz') for p in ens['members']]
    assert all(np.array_equal(m['labels'],gold) for m in members)
    p=np.mean([m['classifier_probabilities'] for m in members],axis=0)
    ensemble_result={'name':ens['name'],'members':ens['members'],'metrics':metrics(p,gold),
                     'paired_vs_round1_supcon87':paired(p,old,gold)}
    np.savez_compressed(ROOT/'ensemble_test_predictions.npz',probabilities=p,labels=gold)
    families={}
    for method,paths in frozen['replicated_families'].items():
        scores=[results[p]['classifier_only']['accuracy'] for p in paths]
        diffs=[]
        for p in paths:
            seed=json.loads((Path(p)/'config.json').read_text())['seed']
            baseline=np.load(f'artifacts/runs/supcon_{seed}/test_predictions.npz')['classifier_probabilities']
            current=np.load(Path(p)/'test_predictions.npz')['classifier_probabilities']
            diffs.append((current.argmax(1)==gold).astype(float)-(baseline.argmax(1)==gold).astype(float))
        mean_delta=np.mean(diffs,axis=0)
        rng=np.random.default_rng(20260921)
        boot=[mean_delta[rng.integers(0,len(gold),len(gold))].mean() for _ in range(10000)]
        families[method]={'accuracy_mean':float(np.mean(scores)),'accuracy_seed_std':float(np.std(scores,ddof=1)),
                         'delta_vs_supcon_mean':float(mean_delta.mean()),'conditional_cluster_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    summary={'freeze':frozen,'single_models':results,'ensemble':ensemble_result,'families':families}
    write_json(ROOT/'test_summary.json',summary)
    print('FINAL',json.dumps(summary),flush=True)


def export():
    frozen=json.loads((ROOT/'evaluation_freeze.json').read_text())
    src=Path(frozen['selected_single']['path']); out=ROOT/'release_single'
    if out.exists(): raise RuntimeError('Release already exists')
    out.mkdir()
    for name in ['model.safetensors','config.json','selection.json','inference_config.json',
                 'validation_metrics.json','test_metrics.json']:
        shutil.copy2(src/name,out/name)
    for sub in ['encoder','tokenizer']:
        shutil.copytree(Path(BASE)/sub,out/'base'/sub)
    for name in ['infer_clinc.py','train_clinc.py','LICENSE']:
        shutil.copy2(name,out/name)
    write_json(out/'provenance.json',{'source':str(src),'selection':'validation only',
                'weight_sha256':digest(out/'model.safetensors'),'round':'2; prior test aggregates known'})
    cfg=json.loads((out/'config.json').read_text())
    score=json.loads((out/'test_metrics.json').read_text())['classifier_only']
    card=f'''# Laya CLINC150 specialist, development round 2

This is a real, fully trained checkpoint of a ModernBERT-based Laya encoder with masked
mean pooling and a 150-class linear head. It is a task-specific derivative, and does
not preserve the original Laya SDK's generic choice/score/noul interface.

- Selected method: `{cfg['method']}`; selected checkpoint: `{src}`.
- Selection: validation accuracy, then validation NLL; test scores were not used to choose this checkpoint.
- Closed-set CLINC150 test accuracy: **{100*score['accuracy']:.4f}%** on 4500 examples.
- Full-data setting: 15000 training, 3000 validation, 4500 test examples; OOS excluded.
- Original base: `convaiinnovations/laya`, revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Dataset: `clinc/oos-eval`, revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`.
- Weights: `model.safetensors`; labels and maximum length are in `config.json`.
- Source-code license: Apache-2.0; upstream license retained.

## Run

This export uses the supplied custom classifier, rather than an AutoModel pipeline:

```bash
python infer_clinc.py --checkpoint . --text "what is my bank balance"
```

Install the dependencies recorded in the experiment's `environment-freeze.txt` /
`requirements.txt`. The export includes encoder configuration and tokenizer under
`base/`; the original pretrained weights are not required for inference.

## Limits

This is self-evaluation, not a verified leaderboard submission. Round-one test aggregate
scores were already known before this development round. No round-two test-error-driven
tuning was performed. The original base model's pretraining mixture is not fully disclosed.
One benchmark does not establish broad transfer, and the employed R-Drop, supervised
contrastive learning, and weight averaging techniques are existing research methods.
See `ROUND2_REPORT.md`, `ROUND2_METHODS.md` and the frozen manifest in the experiment bundle.
'''
    (out/'README.md').write_text(card,encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['freeze','evaluate','export'])
    args=p.parse_args()
    {'freeze':freeze,'evaluate':evaluate,'export':export}[args.action]()
