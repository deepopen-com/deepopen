"""Controlled Laya encoder adaptation: CE versus CE + supervised contrastive loss.

Training reads train/validation only. Test inference is a separate command and
requires a frozen selection.json. The original typed decision heads are replaced
by a 150-class head; this is a specialist derivative, not a general SDK upgrade.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('USE_TF', '0')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from sklearn.metrics import accuracy_score, f1_score, log_loss
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def read_data():
    d = json.loads(Path('data/data_full.json').read_text(encoding='utf-8'))
    labels = sorted({y for _, y in d['train']})
    assert len(labels) == 150
    for split, count in [('train', 15000), ('val', 3000), ('test', 4500)]:
        assert len(d[split]) == count, (split, len(d[split]))
        assert set(y for _, y in d[split]) == set(labels)
    return d, labels


class TextData(Dataset):
    def __init__(self, rows, labels, tok, max_len):
        self.texts = [r[0] for r in rows]
        self.y = torch.tensor([labels.index(r[1]) for r in rows])
        self.tokens = tok(self.texts, padding='max_length', truncation=True,
                          max_length=max_len, return_tensors='pt')
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.tokens['input_ids'][i], self.tokens['attention_mask'][i], self.y[i]


class IntentModel(nn.Module):
    def __init__(self, base, classes=150):
        super().__init__()
        cfg = AutoConfig.from_pretrained(str(Path(base) / 'encoder'), local_files_only=True)
        # Disable optional compilation: reproducible eager path on both GPUs.
        cfg.reference_compile = False
        self.encoder = AutoModel.from_config(cfg, attn_implementation='sdpa')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(cfg.hidden_size, classes)
    def load_laya_encoder(self, base):
        with safe_open(str(Path(base) / 'model.safetensors'), framework='pt', device='cpu') as f:
            weights = {k.removeprefix('encoder.'): f.get_tensor(k)
                       for k in f.keys() if k.startswith('encoder.')}
        self.encoder.load_state_dict(weights, strict=True)
    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        w = mask.to(h.dtype).unsqueeze(-1)
        z = (h * w).sum(1) / w.sum(1).clamp_min(1)
        return self.classifier(self.dropout(z)).float(), F.normalize(z.float(), dim=-1)


def supcon(z, y, temperature=0.1):
    """Supervised contrastive loss, excluding self-pairs and anchors with no positives."""
    n = len(y)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    pos = y[:, None].eq(y[None, :]) & ~eye
    sim = (z @ z.T / temperature).masked_fill(eye, -1e4)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    valid = pos.sum(1) > 0
    if not valid.any():
        return z.sum() * 0
    loss = -(logp * pos).sum(1) / pos.sum(1).clamp_min(1)
    return loss[valid].mean()


@torch.inference_mode()
def infer(model, loader):
    model.eval()
    logits, features, labels = [], [], []
    for ids, mask, y in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l, z = model(ids.cuda(), mask.cuda())
        logits.append(l.cpu()); features.append(z.cpu()); labels.append(y)
    return torch.cat(logits), torch.cat(features), torch.cat(labels)


def metrics(prob, y):
    p = np.asarray(prob)
    y = np.asarray(y)
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == y
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', labels=np.arange(150), zero_division=0)),
            'nll': float(log_loss(y, p, labels=np.arange(150))), 'ece': float(ece), 'n': len(y)}


def make_prototypes(z, y):
    proto = torch.stack([z[y == k].mean(0) for k in range(150)])
    return F.normalize(proto, dim=-1)


def choose_validation(logits, z, y, prototypes):
    """Identical validation search for both methods; never consumes test labels."""
    best = None
    all_rows = []
    for logit_temp in [0.5, 1.0, 2.0]:
        ce = (logits / logit_temp).softmax(-1)
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for proto_temp in ([0.05] if alpha == 0 else [0.02, 0.05, 0.1]):
                pp = (z @ prototypes.T / proto_temp).softmax(-1)
                prob = (1-alpha) * ce + alpha * pp
                m = metrics(prob.numpy(), y.numpy())
                row = {'alpha': alpha, 'logit_temperature': logit_temp,
                       'prototype_temperature': proto_temp, **m}
                all_rows.append(row)
                if best is None or (m['accuracy'], -m['nll']) > (best['accuracy'], -best['nll']):
                    best = row
    return best, all_rows


def train(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'selection.json').exists():
        raise RuntimeError('Completed run exists; use a new output directory')
    seed_all(a.seed)
    torch.set_num_threads(8)
    d, labels = read_data()
    tok = PreTrainedTokenizerFast.from_pretrained(str(Path(a.base) / 'tokenizer'))
    train_ds = TextData(d['train'], labels, tok, a.max_len)
    val_ds = TextData(d['val'], labels, tok, a.max_len)
    loaders = {k: DataLoader(ds, batch_size=a.batch_size, shuffle=k=='train', num_workers=0,
                             pin_memory=True) for k, ds in [('train', train_ds), ('val', val_ds)]}
    train_eval = DataLoader(train_ds, batch_size=a.batch_size, shuffle=False, pin_memory=True)
    model = IntentModel(a.base)
    model.load_laya_encoder(a.base)
    model.cuda()
    opt = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': a.lr},
        {'params': model.classifier.parameters(), 'lr': a.lr * 10}], weight_decay=0.01)
    total = a.epochs * len(loaders['train'])
    warmup = max(1, int(total * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s+1)/warmup, max(0.0, (total-s)/max(1,total-warmup))))
    cfg = vars(a).copy()
    cfg.update(labels=labels, parameters=sum(p.numel() for p in model.parameters()),
               torch=torch.__version__, gpu=torch.cuda.get_device_name(),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/'config.json', cfg)
    best_key = (-1.0, -float('inf'))
    started = time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        losses = []
        for step, (ids, mask, y) in enumerate(loaders['train']):
            opt.zero_grad(set_to_none=True)
            y = y.cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, z = model(ids.cuda(), mask.cuda())
                ce = F.cross_entropy(logits, y)
                con = supcon(z, y) if a.method == 'supcon' else z.sum() * 0
                loss = ce + a.contrastive_weight * con
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                print(json.dumps({'epoch': epoch+1, 'step': step, 'loss': losses[-1],
                                  'seconds': round(time.monotonic()-started,1)}), flush=True)
        vl, vz, vy = infer(model, loaders['val'])
        m = metrics(vl.softmax(-1).numpy(), vy.numpy())
        row = {'epoch': epoch+1, 'train_loss': float(np.mean(losses)), **m,
               'seconds': time.monotonic()-started}
        with open(out/'history.jsonl', 'a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        key = (m['accuracy'], -m['nll'])
        if key > best_key:
            best_key = key
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(out/'model.safetensors'))
            write_json(out/'best_epoch.json', row)
    model.load_state_dict(load_file(str(out/'model.safetensors')))
    _, tz, ty = infer(model, train_eval)
    prototypes = make_prototypes(tz, ty)
    vl, vz, vy = infer(model, loaders['val'])
    selection, grid = choose_validation(vl, vz, vy, prototypes)
    save_file({'prototypes': prototypes}, str(out/'prototypes.safetensors'))
    np.savez_compressed(out/'validation.npz', logits=vl.numpy(), features=vz.numpy(), labels=vy.numpy())
    write_json(out/'validation_search.json', grid)
    selection['training_seconds'] = time.monotonic() - started
    selection['test_used_for_selection'] = False
    write_json(out/'selection.json', selection)
    print('TRAINING_COMPLETE', json.dumps(selection), flush=True)


def evaluate(a):
    out = Path(a.output)
    cfg = json.loads((out/'config.json').read_text())
    selected = json.loads((out/'selection.json').read_text())
    if (out/'test_metrics.json').exists():
        raise RuntimeError('Test results already exist; preserve the first evaluation')
    d, labels = read_data()
    assert labels == cfg['labels']
    tok = PreTrainedTokenizerFast.from_pretrained(str(Path(a.base)/'tokenizer'))
    ds = TextData(d['test'], labels, tok, cfg['max_len'])
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, pin_memory=True)
    model = IntentModel(a.base)
    model.load_state_dict(load_file(str(out/'model.safetensors')), strict=True)
    model.cuda()
    logits, z, y = infer(model, loader)
    proto = load_file(str(out/'prototypes.safetensors'))['prototypes']
    ce = (logits / selected['logit_temperature']).softmax(-1)
    pp = (z @ proto.T / selected['prototype_temperature']).softmax(-1)
    prob = (1-selected['alpha'])*ce + selected['alpha']*pp
    results = {'selected': metrics(prob.numpy(), y.numpy()),
               'classifier_only': metrics(logits.softmax(-1).numpy(), y.numpy()),
               'selection': selected}
    seen = {text.strip().lower() for text, _ in d['train']}
    clean = np.array([text.strip().lower() not in seen for text, _ in d['test']])
    results['selected_excluding_train_duplicates'] = metrics(prob.numpy()[clean], y.numpy()[clean])
    results['classifier_excluding_train_duplicates'] = metrics(logits.softmax(-1).numpy()[clean], y.numpy()[clean])
    np.savez_compressed(out/'test_predictions.npz', probabilities=prob.numpy(),
                        classifier_probabilities=logits.softmax(-1).numpy(), labels=y.numpy())
    with open(out/'test_predictions.jsonl', 'w', encoding='utf-8') as f:
        for i, (text, gold) in enumerate(d['test']):
            f.write(json.dumps({'index': i, 'text': text, 'gold': gold,
                                'prediction': labels[int(prob[i].argmax())],
                                'confidence': float(prob[i].max())}, ensure_ascii=False)+'\n')
    write_json(out/'test_metrics.json', results)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train', 'evaluate'])
    p.add_argument('--base', default='artifacts/base')
    p.add_argument('--output', required=True)
    p.add_argument('--method', choices=['ce', 'supcon'], default='ce')
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-len', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--contrastive-weight', type=float, default=0.1)
    args = p.parse_args()
    (train if args.action == 'train' else evaluate)(args)
