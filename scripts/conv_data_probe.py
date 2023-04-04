from inspect import ArgSpec
import torch
from torch import nn 
import time
import pickle
import torchvision
from perfpred.predictor import Conv2DPredictor
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--use_fp16", action="store_true")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
from perfpred.utils import timing

device = torch.device('cuda')

def load_data(filenames):
    rows = []
    for fn in filenames:
        with open(fn, "rb") as f:
            while 1:
                try:
                    objs = pickle.load(f)
                    rows.extend(objs)
                except (EOFError):
                    break
    return rows

data = load_data([args.filename])
warm_up = 100
nitr = 100

n = 20

err_fw_list = []
err_bw_list = []

dtype = torch.float16 if args.use_fp16 else torch.float32

for i in range(n):
    idx = random.randint(0, len(data)-1)
    while data[idx][5] != 7  or data[idx][8] == 1 or data[idx][0] > 2:
        idx = random.randint(0, len(data)-1)
    d = data[idx]
    dur_fw, dur_bw, _, use_fp16, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = d

    x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, requires_grad=True, dtype=dtype)
    model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    fw_measure = []
    bw_measure = []
    def analyze(prof):
        events = prof.profiler.function_events
        for evt in events:
            if evt.name == "aten::cudnn_convolution":
                fw_measure.append(evt.cuda_time_total/1e3)
            elif evt.name == "aten::convolution_backward":
                bw_measure.append(evt.cuda_time_total/1e3)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    def foo():
        torch.cuda.synchronize()
        optim.zero_grad(set_to_none=True)
        # with torch.autocast(device_type='cuda', dtype=dtype):
        out = model(x)
        loss = out.sum()
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
    for _ in range(warm_up):
        foo()
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=analyze) as prof:
        with torch.autocast(device_type='cuda', dtype=dtype):
            dur_tc = timing(foo, 0, nitr)
    truth_fw = np.mean(fw_measure) 
    truth_bw = np.mean(bw_measure)
    err_fw = (truth_fw - dur_fw) / truth_fw
    err_bw = (truth_bw - dur_bw) / dur_bw
    err_fw_list.append(err_fw)
    err_bw_list.append(err_bw)
    print(f"[{idx}/{len(data)}]\t {np.mean(fw_measure) : .3f}, {np.mean(bw_measure) : .3f}, {dur_fw : .3f}, {dur_bw : .3f},")

print("forward error:", np.mean(err_fw_list))
print("backward errpr:", np.mean(err_bw_list))
