import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torchvision
import subprocess
import time
from perfpred.vgg import build_vgg_model
import argparse
import os

device = os.env("PERPRED_DEVICE")

def get_clock():
    ret = subprocess.run(['nvidia-smi','--query-gpu=clocks.sm', '-i','0','--format=csv,noheader,nounits'], capture_output=True)
    clock = float(ret.stdout)
    return clock

def work():
    device = "cuda"
    model = torchvision.models.vgg13()
    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    nitr = 1000
    for batch_size in [64]:
        x = torch.rand(batch_size, 3, 224, 224, device=device)
        for i in range(nitr):
            optim.zero_grad(set_to_none=True)
            o = model(x)
            o = o.sum()
            o.backward()
            optim.step()
            torch.cuda.synchronize()
        torch.cuda.synchronize()

if __name__ == '__main__':
    p = mp.Process(target=work)
    p.start()
    while p.is_alive():
        print(get_clock())
        time.sleep(0.05)
    p.join()
