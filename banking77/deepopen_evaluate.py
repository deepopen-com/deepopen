"""Evaluate deepopen on the pinned Banking77 public test set."""
import os
os.environ['USE_TF']='0'
import argparse,contextlib,hashlib,json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score,f1_score

REVISION='18072d2685ea682290f7b8924d94c62acc19c0b2'
TEST_SHA256='fb1b0043ded745b8767687084786e6dd0a5f0ce03243b6131992a1c7ae2c2595'
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default=str(Path(__file__).parent))
    p.add_argument('--test-jsonl');p.add_argument('--output',default='evaluation.json')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
    path=Path(a.test_jsonl or hf_hub_download(repo_id='mteb/banking77',repo_type='dataset',revision=REVISION,filename='test.jsonl'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==TEST_SHA256,'Unexpected dataset content'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==3080
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForSequenceClassification.from_pretrained(a.model,attn_implementation='sdpa').to(a.device).eval()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    y=[model.config.label2id[r['label_text']] for r in rows];pred=[]
    assert set(y)==set(range(77)) and all(y.count(k)==40 for k in range(77))
    with torch.inference_mode():
        for start in range(0,len(rows),16):
            b=tok([r['text'] for r in rows[start:start+16]],padding=True,truncation=True,max_length=128,return_tensors='pt').to(a.device)
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if a.device.startswith('cuda') else contextlib.nullcontext()
            with amp:logits=model(**b).logits
            pred.extend(logits.argmax(-1).cpu().tolist())
    metrics={'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'n':len(y)}
    result={'model':'deepopen','dataset_revision':REVISION,'test_sha256':TEST_SHA256,'metrics':metrics,'gold':y,'predicted':pred}
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
