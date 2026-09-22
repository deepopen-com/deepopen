"""Evaluate a saved Laya checkpoint on the untouched official Banking77 test set."""
import argparse,json
from pathlib import Path
from banking77 import ROOT,seed_all,load_agent,make_items,loader,evaluate,dump
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'best_checkpoint'));p.add_argument('--batch',type=int,default=8);p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--output',default=str(ROOT/'evaluation/reproduced.json'));args=p.parse_args()
seed_all(42);data=json.loads((ROOT/'data.json').read_text());agent=load_agent(args.checkpoint)
items=make_items(agent,data[args.split],data['labels']);metrics,gold,pred=evaluate(agent,loader(agent,items,args.batch))
# Check that native SDK inference agrees with the batched evaluation label mapping.
q={'intent':{'type':'choice','instructions':'Which banking intent does `message` express?',
             'criteria':{x.replace('_',' '):None for x in data['labels']}}}
for i,r in enumerate(data[args.split][:20]):
    answer=agent.predict({'message':r['text']},q)['answers']['intent']['choice']
    assert answer==data['labels'][pred[i]].replace('_',' '),(i,answer,pred[i])
dump(args.output,{'split':args.split,'metrics':metrics,'gold':gold,'predicted':pred,'native_sdk_agreement_first20':True})
print(json.dumps(metrics,indent=2))
