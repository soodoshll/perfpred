import torch
import torchvision
import subprocess
import time
import multiprocessing as mp
from perfpred.vgg import build_vgg_model

device = "cuda"
model = torchvision.models.vgg13()
# model = build_vgg_model()
model.to(device)
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

nitr = 1000

def get_clock():
    out = subprocess.run("nvidia-smi -q -d CLOCK -i 0", capture_output=True, shell=True)
    for line in out.stdout.split(b'\n'):
        if line.strip().startswith(b'SM'):
            clock = (int(line.split()[-2]))
            break
    return clock

for batch_size in [64]:
    x = torch.rand(batch_size, 3, 224, 224, device=device)
    t0 = time.time()
    last_t = t0
    for i in range(nitr):
        optim.zero_grad(set_to_none=True)
        o = model(x)
        o = o.sum()
        o.backward()
        optim.step()
        torch.cuda.synchronize()
        if time.time() - last_t >= 10:
            print(f"{batch_size}, {time.time() - t0 :.1f}, {get_clock()}")
            last_t = time.time()
        # if time.time() - t0 >= 65:
            # break
    # assert(not child.is_alive())
    torch.cuda.synchronize()
    