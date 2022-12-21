import torch
from perfpred.utils import timing

device = 'cuda'

for x in range(1, 65, 1):
    def foo():
        a = torch.empty((7, 128, x), device=device, dtype=torch.float16)
        b = torch.empty((7, x, 1024), device=device, dtype=torch.float16)
        out = torch.bmm(a, b)
    t = timing(foo)
    print(f"{x}, {t}")
