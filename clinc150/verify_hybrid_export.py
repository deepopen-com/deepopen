"""Reproduce all frozen test predictions through the actual exported inference API."""
import json
from pathlib import Path
import numpy as np
from infer_hybrid import predict
from train_clinc import write_json

root=Path('artifacts/hybrid');folder=root/'release_system'
data=json.loads(Path('data/data_full.json').read_text());cfg=json.loads((folder/'system.json').read_text());labels=cfg['labels']
gold=np.array([labels.index(y) for _,y in data['test']])
expected=np.load(root/'ensemble_test_predictions.npz') if cfg['selected_type']=='ensemble' else np.load(Path(cfg['source_members'][0])/'test_predictions.npz')
assert np.array_equal(expected['labels'],gold)
results=predict(str(folder),[text for text,_ in data['test']])
actual=np.array([labels.index(r['intent']) for r in results]);reference=expected['fused_probabilities'].argmax(1)
report={'n':len(gold),'actual_export_accuracy':float(np.mean(actual==gold)),'frozen_accuracy':cfg['test_accuracy'],
        'prediction_mismatches':int(np.sum(actual!=reference)),'protocol':'full test inference through exported runtime, same frozen models, no selection or tuning'}
write_json(root/'export_reproduction.json',report)
np.savez_compressed(root/'export_predictions.npz',gold=gold.astype(np.int16),prediction=actual.astype(np.int16))
print(json.dumps(report),flush=True)
assert report['prediction_mismatches']==0,report
assert abs(report['actual_export_accuracy']-cfg['test_accuracy'])<1e-12
