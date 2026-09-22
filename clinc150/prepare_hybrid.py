"""Pin an official DeBERTa control and verify mirror downloads by upstream LFS hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODEL='microsoft/deberta-v3-large';REV='64a8c8eab3e352a784c658aef62be1662607476f'
FILES=['config.json','tokenizer_config.json','spm.model','pytorch_model.bin','README.md']
ROOT=Path('artifacts/hybrid');BASE=ROOT/'base'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main(metadata_only):
    ROOT.mkdir(parents=True,exist_ok=True)
    if metadata_only:
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/revision/{REV}?blobs=true',timeout=40) as r:d=json.load(r)
        assert d['sha']==REV
        entries={r['rfilename']:r for r in d['siblings'] if r['rfilename'] in FILES}
        (ROOT/'upstream_manifest.json').write_text(json.dumps({'model':MODEL,'revision':REV,'files':entries},indent=2))
        print(json.dumps(entries),flush=True);return
    import requests
    expected=json.loads((ROOT/'upstream_manifest.json').read_text());BASE.mkdir(exist_ok=True)
    for name in FILES:
        path=BASE/name
        if not path.exists():
            url=f'https://hf-mirror.com/{MODEL}/resolve/{REV}/{name}'
            r=requests.get(url,stream=True,timeout=60)
            if r.status_code==403:
                r.close();r=requests.get(f'https://hf-mirror.com/{MODEL}/resolve/main/{name}',stream=True,timeout=60)
            r.raise_for_status()
            with open(path.with_suffix(path.suffix+'.part'),'wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
            path.with_suffix(path.suffix+'.part').replace(path)
        info=expected['files'][name]
        assert path.stat().st_size==info['size'],name
        if 'lfs' in info:assert sha(path)==info['lfs']['sha256'],name
        else:
            body=path.read_bytes();assert hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()==info['blobId'],name
        print('VERIFIED',name,path.stat().st_size,sha(path),flush=True)
    (ROOT/'download_manifest.json').write_text(json.dumps({'model':MODEL,'revision':REV,'files':{n:{'sha256':sha(BASE/n),'bytes':(BASE/n).stat().st_size} for n in FILES}},indent=2))
    (ROOT/'BASE_READY').touch()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata-only',action='store_true');a=p.parse_args();main(a.metadata_only)
