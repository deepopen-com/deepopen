"""Download pinned, public CLINC150 data and the English Laya checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import urllib.error

DATA_REV = '828f8093932c8fe6ca7936c3d2e52903b1c523de'
MODEL_REV = '1c5edc17a7acd8701df6fc341c0d179f1c62c982'
FILES = ['rl_agent_config.json', 'encoder/config.json', 'tokenizer/tokenizer.json',
         'tokenizer/tokenizer_config.json', 'model.safetensors']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print('Downloading', url, flush=True)
    tmp = path.with_suffix(path.suffix + '.part')
    req = urllib.request.Request(url, headers={'User-Agent': 'python-requests/2.32.5'})
    try:
        response = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code != 403 or 'hf-mirror.com' not in url:
            raise
        # Some mirrors only serve the main resolve URL. The expected manifest
        # below must verify every byte before these files can be used.
        if not Path('artifacts/expected_manifest.json').exists():
            raise RuntimeError('Mirror fallback requires the independently downloaded manifest')
        mirror_url = url.replace('/resolve/' + MODEL_REV + '/', '/resolve/main/')
        response = urllib.request.urlopen(urllib.request.Request(mirror_url,
            headers={'User-Agent':'python-requests/2.32.5'}), timeout=60)
    with response as r, open(tmp, 'wb') as f:
        while chunk := r.read(8 * 1024 * 1024):
            f.write(chunk)
    tmp.replace(path)


def main():
    fetch(f'https://raw.githubusercontent.com/clinc/oos-eval/{DATA_REV}/data/data_full.json',
          Path('data/data_full.json'))
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    for file in FILES:
        fetch(f'{endpoint}/convaiinnovations/laya/resolve/{MODEL_REV}/{file}', Path('artifacts/base') / file)
    manifest = {'data_revision': DATA_REV, 'model_revision': MODEL_REV,
                'model_id': 'convaiinnovations/laya', 'files': {}}
    for p in [Path('data/data_full.json')] + [Path('artifacts/base') / x for x in FILES]:
        manifest['files'][str(p)] = {'sha256': sha256(p), 'bytes': p.stat().st_size}
    expected = Path('artifacts/expected_manifest.json')
    if expected.exists():
        ref = json.loads(expected.read_text(encoding='utf-8'))
        for name, item in ref['files'].items():
            normalized = name.replace('\\', '/')
            if manifest['files'][normalized] != item:
                raise RuntimeError('Downloaded file does not match pinned checkpoint: '+normalized)
    Path('artifacts/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Verified manifest written.', flush=True)


if __name__ == '__main__':
    main()
