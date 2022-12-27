import torch
from torch import nn
import os
import argparse
import time
import torchvision
from torch.autograd import DeviceType
from perfpred.measure import get_children_kernel_time2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('batch_size', type=int)
parser.add_argument('--model', type=str, default="vgg13")
parser.add_argument('--amp', action='store_true')
args = parser.parse_args()

model = getattr(torchvision.models, args.model)()
device = torch.device('cuda')
model.to(device)

def _get_all_children(events, root):
    ret = []
    for evt in events:
        if evt.time_range.start >= root.time_range.start and evt.time_range.end <= root.time_range.end:
            ret.append(evt)
    return ret

def _get_children_kernel_time(event, marked_kernel=None):
    kernel_time = sum([kernel.duration for kernel in event.kernels])
    if marked_kernel is not None:
        marked_kernel.update(event.kernels)
    for child in event.cpu_children:
        kernel_time += _get_children_kernel_time(child, marked_kernel)
    return kernel_time

def analyze_profile(prof):
    events = prof.profiler.function_events
    cnt = 0
    for evt in events:
        if evt.name == 'ProfilerStep*':
            cnt += 1
            if cnt == 0:
                continue
            children = _get_all_children(events, evt)
            gpu_time = 0
            cpu_time = total_time = evt.cpu_time_total
            # for child in evt.cpu_children:
                # print(child.name)
            for child in children:
                # print(child.name)
                if child.name.startswith("cudaDeviceSynchronize") or child.name == 'cudaMemcpyAsync':
                    cpu_time -= child.cpu_time_total
                kernel_time = sum([kernel.duration for kernel in child.kernels])
                gpu_time += kernel_time
            total_times.append(total_time)
            gpu_times.append(gpu_time)
            cpu_times.append(cpu_time)

bs = args.batch_size
image_size = 224
x = torch.rand((bs, 3, image_size, image_size), device=device)
scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

cpu_times = []
gpu_times = []
total_times = []

t = torch.randint(1000, (bs, ), device=device)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_loop():
    optim.zero_grad(set_to_none=False)
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
        out = model(x)
        loss = loss_fn(out, t)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    torch.cuda.synchronize()

with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1),
    on_trace_ready=analyze_profile,
    with_stack=True
) as profiler:
    for i in range(15):
        train_loop()
        profiler.step()
profiler.export_chrome_trace('trace_amp.json')
print(f"{np.mean(cpu_times)/1e3 : .2f}, {np.mean(gpu_times)/1e3 : .2f}, {np.mean(total_times)/1e3 :.2f}")