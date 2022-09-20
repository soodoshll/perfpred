from inspect import ArgSpec
import torch
from torch import nn 
import time
# from apex import amp 
import pickle
import torchvision
from perfpred.predictor import load_model, Conv2DPredictor
from perfpred.utils import get_clock
from perfpred.vgg import build_vgg_model
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from tqdm import tqdm
import random
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_fp16", action="store_true")
parser.add_argument("--clock", action="store_true")
parser.add_argument("--cooldown", type=float, default=0)
parser.add_argument("--period", type=int, default=1)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
# linear_pred, conv_pred, maxpool_pred = load_model()
from perfpred.utils import timing

device = torch.device('cuda')

err_fw_list = []
err_bw_list = []

dtype = torch.float16 if args.use_fp16 else torch.float32

use_fp16, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = False, 16, 3, 224, 64, 64, 1, 1

x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, requires_grad=True, dtype=dtype)
model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
# model = build_vgg_model(False)
model.to(device)
# model(x)
fw_measure = []
bw_measure = []
clock_list = []
n = 1_000 
def analyze(prof):
    events = prof.profiler.function_events
    for evt in events:
        if evt.name == "aten::cudnn_convolution":
            fw_measure.append(evt.cuda_time_total/1e3)
            # print(evt.cuda_time_total/1e3)
        # elif evt.name == "aten::convolution_backward":
            # bw_measure.append(evt.cuda_time_total/1e3)
optim = torch.optim.SGD(model.parameters(), lr=1e-3)
def foo():
    optim.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=dtype):
        out = model(x)
        loss = out.sum()
        loss.backward()
    optim.step()
    return out
    # time.sleep(0.1)
with profile(activities=[
    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
     on_trace_ready=analyze
     ) as prof:
    with torch.autocast(device_type='cuda', dtype=dtype):
        for i in tqdm(range(n)):
            torch.cuda.synchronize()
            t0 = time.time()
            foo()
            # torch.cuda.synchronize()
            # if i % 100 == 99:
            dur = time.time() - t0
            if args.clock and (i % 10 == 1):
                clock_list.append(get_clock())
                # print(get_clock())
            torch.cuda.synchronize()
            # if i % 30 == 0:
            if i % args.period == 0:
                time.sleep(args.cooldown)
# prof.export_chrome_trace("trace.json")
            # fw_measure.append(dur)
# print(batch_size, dur, dur_tc)
truth_fw = np.mean(fw_measure) 
clock_list = np.array(clock_list)
# truth_bw = np.mean(bw_measure)
np.savez("data/conv_long_time.npz", fw_measure=fw_measure, clock=clock_list)

avg = np.mean(fw_measure[1:])
std = np.std(fw_measure[1:])
print(f"avg: {avg :.2f} | std: {std :.2f}")

# data = np.load('data/conv_long_time.npz')

# fw_measure = data['fw_measure']
# clock_list = data['clock']
if args.plot:
    plt.subplots(figsize=(12, 12))
    plt.scatter(range(1, n), fw_measure[1:], marker=".", alpha=0.1)
    if args.clock:
        ax = plt.twinx()
        ax.plot(range(1, n, 10), clock_list, 'r.', linewidth=1, alpha=0.1)
    plt.savefig("figure/conv_longtime.png", )