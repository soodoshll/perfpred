import torch
import numpy as np
device = torch.device("cuda")

pool = []
size = 1024 
tot_size = 0

linear = torch.nn.Linear(1024, 1024, device=device, dtype=float)

for i in range(10000):
    tensor = torch.rand((size, size), dtype=float, device=device)
    out = linear(tensor)
    pool.append(out)
    tot_size += size * size * 4
    print(f"allocated: {tot_size/1e9 : .2f} GB", )
torch.cuda.synchronize()