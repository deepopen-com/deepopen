"""Check balanced coverage and symmetric-KL invariants before spending GPU time."""
import numpy as np
import torch
from round2 import BalancedBatches, symmetric_kl

y=np.repeat(np.arange(150),100)
sampler=BalancedBatches(y,13)
batches=list(sampler)
assert len(batches)==len(sampler)==235
assert sorted(i for b in batches for i in b)==list(range(15000))
assert all(0<len(b)<=64 for b in batches)
assert all(all(n>=4 and n%4==0 for n in np.unique(y[b],return_counts=True)[1]) for b in batches)
assert batches!=list(sampler)
a=torch.randn(8,150,requires_grad=True)
b=torch.randn(8,150,requires_grad=True)
assert abs(symmetric_kl(a,a).item())<1e-6
assert torch.allclose(symmetric_kl(a,b),symmetric_kl(b,a))
loss=symmetric_kl(a,b)
assert loss.item()>0
loss.backward()
assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
print('PASS: exactly-once balanced coverage, positive pairs, symmetric KL gradients')
