import torch
from perfpred.utils import timing

device = 'cuda'

for bs in range(1, 32):
    def foo():
        a = torch.empty((bs, 128, 1024), device=device, dtype=torch.float16)
        b = torch.empty((bs, 1024, 1024), device=device, dtype=torch.float16)
        out = torch.bmm(a, b)
    t = timing(foo)
    print(f"{bs}, {t}")