"""Scientific invariants and input-budget audit; no training required."""
import json
from pathlib import Path
import sys
import unittest
import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast
from train_clinc import supcon, make_prototypes, read_data, write_json


class LossTests(unittest.TestCase):
    def test_no_positive_is_finite_and_differentiable(self):
        z = torch.randn(4, 8, requires_grad=True)
        loss = supcon(F.normalize(z, dim=-1), torch.arange(4))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_clusters_have_lower_loss_than_mixed_pairs(self):
        z = torch.tensor([[1.,0.], [1.,0.], [0.,1.], [0.,1.]])
        self.assertLess(supcon(z, torch.tensor([0,0,1,1])),
                        supcon(z, torch.tensor([0,1,0,1])))

    def test_permutation_invariance(self):
        torch.manual_seed(19)
        z = F.normalize(torch.randn(8, 12), dim=-1)
        y = torch.tensor([0,0,1,1,2,2,3,3])
        p = torch.randperm(8)
        self.assertTrue(torch.allclose(supcon(z,y), supcon(z[p],y[p]), atol=1e-6))


def audit():
    d, labels = read_data()
    tok = PreTrainedTokenizerFast.from_pretrained('artifacts/base/tokenizer')
    sys.path.insert(0, str(Path('upstream').resolve()))
    from laya.common import build_sequence
    q = {'t':'choice', 'ins':'Identify the intent of this user request.',
         'crit':{label:label.replace('_',' ') for label in labels}}
    ids, markers = build_sequence(tok, d['train'][0][0], q)
    splits = {s:{t.strip().lower() for t,_ in d[s]} for s in ['train','val','test']}
    lengths = {s:[len(x) for x in tok([t for t,_ in d[s]], add_special_tokens=True)['input_ids']]
               for s in ['train','val']}
    result = {'expected_classes':150, 'original_default_marker_count':len(markers),
              'original_sequence_length':len(ids),
              'original_full_150_representable':len(markers)==150,
              'normalized_text_overlap':{a+'_'+b:len(splits[a]&splits[b])
                  for a,b in [('train','val'),('train','test'),('val','test')]},
              'lengths':{s:{'max':max(v), 'over_64':sum(n>64 for n in v), 'count':len(v)}
                         for s,v in lengths.items()},
              'upstream_training_contamination':'Unknown: public Laya training mixture is not fully audited.'}
    write_json('artifacts/data_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LossTests)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit(1)
    audit()
