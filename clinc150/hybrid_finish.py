import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from transformers import AutoTokenizer
from hybrid import ROOT,BASE,LAYA,HybridControl,choose
from round3 import digest
from train_clinc import TextData,infer,metrics,write_json
from round2_finish import paired

def freeze():
    assert (ROOT/'TRAINING_COMPLETE').exists() and not (ROOT/'evaluation_freeze.json').exists()
    paths=[str(p) for p in sorted(ROOT.glob('deberta_*')) if (p/'COMPLETE').exists()]
    old=np.load(Path(LAYA)/'validation.npz');reference=torch.from_numpy(old['logits']).softmax(-1).numpy();gold=old['labels']
    rows=[];prob=[]
    for p in paths:
        r=np.load(Path(p)/'validation.npz');assert np.array_equal(r['labels'],gold);prob.append(r['probabilities'])
        rows.append({'path':p,**json.loads((Path(p)/'selection.json').read_text())})
    selected=max(rows,key=lambda r:(r['accuracy'],-r['nll']));ensemble,grid=choose(np.mean(prob,axis=0),reference,gold)
    overall='ensemble' if len(paths)>1 and (ensemble['accuracy'],-ensemble['nll'])>(selected['accuracy'],-selected['nll']) else 'single'
    sources=['hybrid.py','hybrid_finish.py','hybrid_replicates.py','infer_hybrid.py','HYBRID_PLAN.md','prepare_hybrid.py','train_clinc.py','round2.py','round2_finish.py','round3.py']
    f={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'prior_test_aggregates_known':True,'paths':paths,'all_validation':rows,
       'selected_single':selected,'ensemble_selection':ensemble,'ensemble_grid':grid,'overall_type':overall,
       'source_sha256':{p:digest(p) for p in sources},'weight_sha256':{p:digest(Path(p)/'model.safetensors') for p in paths+[LAYA]},
       'selection_sha256':{p:digest(Path(p)/'selection.json') for p in paths},'data_sha256':digest('data/data_full.json'),'base_manifest_sha256':digest(ROOT/'download_manifest.json')}
    write_json(ROOT/'evaluation_freeze.json',f);print(json.dumps(f),flush=True)

def evaluate():
    f=json.loads((ROOT/'evaluation_freeze.json').read_text());assert not (ROOT/'test_summary.json').exists();torch.set_num_threads(6)
    for p,sha in f['source_sha256'].items():assert digest(p)==sha,p
    for p,sha in f['weight_sha256'].items():assert digest(Path(p)/'model.safetensors')==sha,p
    for p,sha in f['selection_sha256'].items():assert digest(Path(p)/'selection.json')==sha,p
    assert digest('data/data_full.json')==f['data_sha256'];assert digest(ROOT/'download_manifest.json')==f['base_manifest_sha256']
    data=json.loads(Path('data/data_full.json').read_text());labels=sorted({y for _,y in data['train']})
    tok=AutoTokenizer.from_pretrained(BASE,use_fast=False,local_files_only=True)
    loader=DataLoader(TextData(data['test'],labels,tok,64),batch_size=32)
    reference=np.load(Path(LAYA)/'test_predictions.npz');gold=reference['labels'];old=reference['classifier_probabilities'];results={};prob=[]
    for path in f['paths']:
        out=Path(path);assert not (out/'test_metrics.json').exists()
        model=HybridControl(pretrained=False);model.load_state_dict(load_file(str(out/'model.safetensors')),strict=True);model.cuda()
        l,_,y=infer(model,loader);p=l.softmax(-1).numpy();assert np.array_equal(y.numpy(),gold)
        selection=json.loads((out/'selection.json').read_text());a=selection['laya_weight'];mixed=(1-a)*p+a*old
        m={'deberta_component':metrics(p,gold),'selected_fusion':metrics(mixed,gold),'paired_vs_laya':paired(mixed,old,gold),'paired_vs_deberta_component':paired(mixed,p,gold),'laya_weight':a}
        write_json(out/'test_metrics.json',m);np.savez_compressed(out/'test_predictions.npz',probabilities=p,fused_probabilities=mixed,labels=gold)
        results[path]=m;prob.append(p);del model;torch.cuda.empty_cache();print(path,json.dumps(m),flush=True)
    p=np.mean(prob,axis=0);a=f['ensemble_selection']['laya_weight'];mixed=(1-a)*p+a*old
    ensemble={'deberta_component':metrics(p,gold),'selected_fusion':metrics(mixed,gold),'paired_vs_laya':paired(mixed,old,gold),'paired_vs_deberta_component':paired(mixed,p,gold),'laya_weight':a}
    np.savez_compressed(ROOT/'ensemble_test_predictions.npz',probabilities=p,fused_probabilities=mixed,labels=gold)
    chosen=ensemble if f['overall_type']=='ensemble' else results[f['selected_single']['path']]
    scores=[r['selected_fusion']['accuracy'] for r in results.values()]
    summary={'freeze':f,'models':results,'ensemble':ensemble,'selected_result':chosen,
             'family_mean':float(np.mean(scores)),'family_seed_std':float(np.std(scores,ddof=1)) if len(scores)>1 else None,
             'component_checkpoint_selection':'Each DeBERTa checkpoint was selected by validation fusion accuracy/NLL; standalone component numbers do not claim separately optimized DeBERTa baselines.'}
    write_json(ROOT/'test_summary.json',summary);print('FINAL',json.dumps(summary),flush=True)

def export():
    s=json.loads((ROOT/'test_summary.json').read_text());f=s['freeze'];out=ROOT/'release_system';assert not out.exists();out.mkdir()
    members=f['paths'] if f['overall_type']=='ensemble' else [f['selected_single']['path']]
    alpha=s['selected_result']['laya_weight'];deployed=[]
    if alpha<1:
        for i,path in enumerate(members):
            name=f'deberta_{i}';dest=out/name;dest.mkdir()
            for file in ['model.safetensors','config.json','selection.json']:shutil.copy2(Path(path)/file,dest/file)
            assert digest(dest/'model.safetensors')==f['weight_sha256'][path];deployed.append(name)
        (out/'deberta_base').mkdir()
        for name in ['config.json','tokenizer_config.json','spm.model','README.md']:shutil.copy2(BASE/name,out/'deberta_base'/name)
    if alpha>0:
        (out/'laya').mkdir()
        for name in ['model.safetensors','config.json','selection.json','inference_config.json']:shutil.copy2(Path(LAYA)/name,out/'laya'/name)
        for sub in ['encoder','tokenizer']:shutil.copytree(Path('artifacts/base')/sub,out/'laya/base'/sub)
        assert digest(out/'laya/model.safetensors')==f['weight_sha256'][LAYA]
    labels=json.loads((Path(members[0])/'config.json').read_text())['labels']
    write_json(out/'system.json',{'deberta_members':deployed,'laya_weight':alpha,'labels':labels,'selected_type':f['overall_type'],
                                'test_accuracy':s['selected_result']['selected_fusion']['accuracy'],'source_members':members,'weight_sha256':f['weight_sha256'],
                                'is_laya_hybrid':0<alpha<1})
    for name in ['infer_hybrid.py','hybrid.py','train_clinc.py','round2.py','round3.py','infer_clinc.py','LICENSE']:shutil.copy2(name,out/name)
    (out/'README.md').write_text(f'''# Heterogeneous intent classification system

Selected test accuracy: {100*s['selected_result']['selected_fusion']['accuracy']:.4f}% on CLINC150 full closed-set test.
Laya probability weight: {alpha}; DeBERTa members deployed: {len(deployed)}.
If Laya weight is zero, this is a DeBERTa control, NOT a Laya improvement.
Run from this directory: `python infer_hybrid.py --checkpoint . --text "what is my bank balance"`.
All required inference weights and tokenizers are included. Dataset training counts remain 15000/3000/4500.
This is subsequent benchmark development, not an independent blind test or leaderboard acceptance.
DeBERTa-v3-large official model card declares MIT; its original model card is retained. Local code retains Apache-2.0 LICENSE.
''',encoding='utf-8');print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate','export']);a=p.parse_args();{'freeze':freeze,'evaluate':evaluate,'export':export}[a.action]()
