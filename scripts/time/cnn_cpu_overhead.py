import torch
from torch import nn 
import time
import torchvision
from perfpred.utils import get_clock
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_fp16", action="store_true")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
# linear_pred, conv_pred, maxpool_pred = load_model()
from perfpred.utils import timing

device = torch.device('cuda')
dtype = torch.float16 if args.use_fp16 else torch.float32

n = 20

def get_all_children(events, root):
    ret = []
    for evt in events:
        if evt.time_range.start >= root.time_range.start and evt.time_range.end <= root.time_range.end:
            ret.append(evt.name)
    return ret

def find_next(events, eid):
    for i in range(eid, len(events)):
        if events[i].time_range.start >= events[eid].time_range.end:
            return events[i]

def func(model, inputs):
    fw_measure = []
    def analyze(prof):
        ret = []
        events = prof.profiler.function_events
        for eid, evt in enumerate(events):
            if evt.name == "aten::conv2d":
                # children = get_all_children(events, evt)
                real_dur = find_next(events, eid).time_range.start - evt.time_range.start
                # print(evt.cpu_parent, evt.cpu_children)
                fw_measure.append(real_dur/1e3)
        # fw_measure.append(ret)

    def foo():
        with torch.no_grad():
            model(inputs)
        torch.cuda.synchronize()
    foo()
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
        on_trace_ready=analyze
        ) as prof:
        with torch.autocast(device_type='cuda', dtype=dtype):
            for i in range(n):
                foo()
    # prof.export_chrome_trace('trace_cpu_resnet.json')
    return fw_measure

inputs = torch.empty((8, 3, 224, 224), device=device)
models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
models += ['densenet121', 'densenet161', 'densenet169', 'densenet201']
models += ['vgg11', 'vgg13', 'vgg16', 'vgg19']
models += ['mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenet_v2']
models += ['resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d']
models += ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
models += ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l']
models += ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']
models += ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf']
models += ['regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf', 'regnet_y_128gf']
models += ['regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf']
models += ['regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf']
models += ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
models += ['wide_resnet50_2', 'wide_resnet101_2']
# models = ['resnet18']
for model_name in models:
    model = getattr(torchvision.models, model_name)()
    model.to(device)
    ret = func(model, inputs)
    print(len(ret)/n, sum(ret)/n)