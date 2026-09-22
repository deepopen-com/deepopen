import os
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, json, time, hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from banking77 import ROOT, seed_all, dump
from prototype_model import PrototypeClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'evaluate', 'smoke'])
    p.add_argument('--variant', choices=['control', 'prototype', 'semantic'], default='semantic')
    p.add_argument('--name', default='semantic_s42')
    p.add_argument('--source', default=str(ROOT / 'classifier_runs/encoder_lr2e5/best'))
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--full-train', action='store_true')
    p.add_argument('--checkpoint')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = p.parse_args()
    seed_all(args.seed)
    data = json.loads((ROOT / 'data.json').read_text())
    out = ROOT / 'research_v2' / args.name
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(Path(args.checkpoint) / 'backbone' if args.checkpoint else args.source)
    collator = DataCollatorWithPadding(tok)

    def batches(split, shuffle=False):
        rows = data[split]
        enc = tok([x['text'] for x in rows], max_length=128, truncation=True)
        items = [{**{k: v[i] for k, v in enc.items()}, 'labels': x['label']} for i, x in enumerate(rows)]
        return DataLoader(items, batch_size=16, shuffle=shuffle, collate_fn=collator, pin_memory=True)

    @torch.inference_mode()
    def evaluate(model, split):
        model.eval(); gold = []; pred = []
        for b in batches(split):
            gold.extend(b.pop('labels').tolist())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(**{k: v.cuda() for k, v in b.items()})['logits']
            assert torch.isfinite(logits).all()
            pred.extend(logits.argmax(-1).cpu().tolist())
        return {'accuracy': accuracy_score(gold, pred), 'macro_f1': f1_score(gold, pred, average='macro'), 'n': len(gold)}, gold, pred

    if args.mode == 'evaluate':
        model = PrototypeClassifier.from_pretrained(args.checkpoint).cuda()
        assert [model.backbone.config.id2label[i] for i in range(77)] == data['labels']
        metrics, gold, pred = evaluate(model, args.split)
        dump(out / (args.split + '_predictions.json'), {'metrics': metrics, 'gold': gold, 'predicted': pred, 'checkpoint': args.checkpoint})
        dump(out / (args.split + '_report.json'), classification_report(gold, pred, target_names=data['labels'], output_dict=True))
        print(json.dumps(metrics), flush=True)
        return

    # Validation experiments must never start from a model trained on full_train.
    if not args.full_train:
        assert Path(args.source).resolve() == (ROOT / 'classifier_runs/encoder_lr2e5/best').resolve()
    backbone = AutoModelForSequenceClassification.from_pretrained(args.source, attn_implementation='sdpa').cuda()
    train_split = 'full_train' if args.full_train else 'train'
    config = {**vars(args), 'train_split': train_split, 'effective_batch': 32, 'micro_batch': 16, 'weight_decay': .01, 'warmup_ratio': .1, 'head_lr_multiplier': 10, 'semantic_fraction': .2, 'test_previously_exposed': True, 'data_sha256': hashlib.sha256((ROOT / 'data.json').read_bytes()).hexdigest()}
    dump(out / 'experiment_config.json', config)
    d = backbone.config.hidden_size
    anchors = torch.zeros(77, d, device='cuda')
    if args.variant != 'control':
        # DataLoader iteration consumes CPU RNG even without shuffling. Keep
        # cache creation vs reuse from changing the subsequent training order.
        rng_before_anchors = torch.get_rng_state()
        cache = ROOT / 'research_v2' / ('anchors_full.pt' if args.full_train else 'anchors_train.pt')
        # Both caches are created only from their declared training split, never test/validation.
        if cache.exists():
            saved = torch.load(cache, weights_only=True, map_location='cuda')
            assert saved['source'] == str(Path(args.source).resolve()) and saved['data_sha256'] == config['data_sha256']
            centroid, semantic = saved['centroid'], saved['semantic']
        else:
            counts = torch.zeros(77, device='cuda')
            backbone.eval()
            with torch.inference_mode():
                for b in batches(train_split):
                    y = b.pop('labels').cuda(); b = {k: v.cuda() for k, v in b.items()}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        h = backbone.model(**b).last_hidden_state
                    w = b['attention_mask'].unsqueeze(-1)
                    z = F.normalize(((h.float() * w).sum(1) / w.sum(1)), dim=-1)
                    anchors.index_add_(0, y, z); counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
                assert (counts > 0).all()
                centroid = F.normalize(anchors / counts[:, None], dim=-1)
                label_text = ['A banking customer asks about ' + s.replace('_', ' ') + '.' for s in data['labels']]
                b = tok(label_text, padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    h = backbone.model(**b).last_hidden_state
                w = b['attention_mask'].unsqueeze(-1)
                semantic = F.normalize((h.float() * w).sum(1) / w.sum(1), dim=-1)
            torch.save({'centroid': centroid.cpu(), 'semantic': semantic.cpu(), 'source': str(Path(args.source).resolve()), 'data_sha256': config['data_sha256']}, cache)
        anchors = F.normalize(.8 * centroid + .2 * semantic, dim=-1) if args.variant == 'semantic' else centroid
        torch.set_rng_state(rng_before_anchors)
    model = PrototypeClassifier(backbone, args.variant, anchors.cpu()).cuda()
    train = batches(train_split, True)
    if args.mode == 'smoke':
        b = {k: v.cuda() for k, v in next(iter(train)).items()}
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(**b)
            ref = backbone(**b).logits
            assert torch.allclose(result['base_logits'], ref, atol=.02, rtol=.02)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**b)['loss']
        loss.backward()
        assert torch.isfinite(loss) and model.prototypes.grad is not None
        assert torch.isfinite(model.prototypes.grad).all()
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected = model(**b)['logits'].cpu()
        model.save_pretrained(out / 'smoke_checkpoint'); tok.save_pretrained(out / 'smoke_checkpoint/backbone')
        restored = PrototypeClassifier.from_pretrained(out / 'smoke_checkpoint').cuda().eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            actual = restored(**b)['logits'].cpu()
        assert torch.equal(expected.argmax(-1), actual.argmax(-1))
        assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
        print('SMOKE_PASS: original logits match; prototype gradients finite; save/load logits identical', flush=True)
        return
    groups = []
    for enc in [True, False]:
        for decay in [True, False]:
            params = [v for n, v in model.named_parameters() if n.startswith('backbone.model.') == enc and (v.ndim >= 2) == decay]
            groups.append({'params': params, 'lr': args.lr * (1 if enc else 10), 'weight_decay': .01 if decay else 0.})
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = (len(train) + 1) // 2
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps_per_epoch * args.epochs * .1), steps_per_epoch * args.epochs)
    history = []; best = (-1., -1.); start = time.time()
    if not args.full_train:
        initial, _, _ = evaluate(model, 'validation'); dump(out / 'initial.json', initial)
        print('INITIAL ' + json.dumps(initial), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.; count = 0; optimizer.zero_grad(set_to_none=True)
        # Accumulate sample-weighted loss, including the final incomplete batch.
        for step, b in enumerate(train):
            n = len(b['labels']); group_start = (step // 2) * 32
            denom = min(32, len(train.dataset) - group_start)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**{k: v.cuda(non_blocking=True) for k, v in b.items()})['loss']
            assert torch.isfinite(loss)
            (loss * n / denom).backward(); total += float(loss.detach()) * n; count += n
            if step % 2 == 1 or step == len(train) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        metrics = {} if args.full_train else evaluate(model, 'validation')[0]
        metrics.update(epoch=epoch, loss=total/count, seconds=time.time()-start, gate=float(model.gate.sigmoid().detach()), scale=float(model.log_scale.exp().detach()))
        history.append(metrics); dump(out / 'history.json', history); print(json.dumps(metrics), flush=True)
        score = (metrics.get('accuracy', -1), metrics.get('macro_f1', -1))
        if (args.full_train and epoch == args.epochs) or (not args.full_train and score > best):
            best = score
            model.save_pretrained(out / 'best'); tok.save_pretrained(out / 'best/backbone')
            dump(out / 'best.json', metrics)
    dump(out / 'done.json', {'best': json.loads((out/'best.json').read_text()), 'seconds': time.time()-start})


if __name__ == '__main__':
    main()
