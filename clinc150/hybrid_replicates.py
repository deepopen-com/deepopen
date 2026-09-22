import json
from pathlib import Path
root=Path('artifacts/hybrid')
assert (root/'PILOT_COMPLETE').exists()
assert not (root/'replicate_decision.json').exists()
s=json.loads((root/'deberta_42/selection.json').read_text())
decision={'replicate':s['accuracy']>.984+1e-12,'pilot_selection':s,'rule':'selected validation accuracy > 0.984','test_used':False}
(root/'replicate_decision.json').write_text(json.dumps(decision,indent=2))
print('yes' if decision['replicate'] else 'no')
