from inspect import ArgSpec
import torch
from torch import nn 
import time
from vgg import build_vgg_model
# from apex import amp 
import pickle
import torchvision
from perfpred.predictor import load_model, Conv2DPredictor
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
# linear_pred, conv_pred, maxpool_pred = load_model()
conv_pred = Conv2DPredictor(True)
conv_pred.load_model("./model/predictor_model_conv2d.th")

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

# data = load_data(["data/conv_2080ti_fp16_0.data"])
data = load_data([args.filename])
# data = data[-30:]
# print(data)
# batch_size = 32
# in_channels = 64
# image_size = 224
warm_up = 1
nitr = 3

# out_channels = 64
# model = nn.Conv2d(in_channels, out_channels, 3, bias=False)
# # model = torchvision.models.resnet18()
# model.to(device)
# x = torch.rand((batch_size, in_channels, image_size, image_size), device=device)
# model(x)

# # model, optim = amp.initialize(model, optim, opt_level="O3")
# scaler = torch.cuda.amp.GradScaler()

# scaler = torch.cuda.amp.GradScaler()
# for d in data:
n = 20

err_fw_list = []
err_bw_list = []
for i in range(n):
    idx = random.randint(0, len(data)-1)
    while data[idx][5] != 7 :
        idx = random.randint(0, len(data)-1)
    d = data[idx]
    dur_fw, dur_bw, _, use_fp16, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = d
    # print(d)
    pred_fw = conv_pred.predict(
        [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 0]
    )    
    pred_bw = conv_pred.predict(
        [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 0, 0]
    )    
    x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, requires_grad=True)
    model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    # model(x)
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
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = model(x)
        loss = out.sum()
        loss.backward()
        # scaler.scale(loss).backward()
        # scaler.step(optim)
        # scaler.update()
        # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
            # scaled_loss.backward()
        optim.step()
        torch.cuda.synchronize()
    for _ in range(warm_up):
        foo()
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=analyze) as prof:
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        # dur = timing(lambda: model(x), nitr)
        dur_tc = timing(foo, warm_up, nitr)
    # print(batch_size, dur, dur_tc)
    truth_fw = np.mean(fw_measure) 
    truth_bw = np.mean(bw_measure)
    err_fw = (truth_fw - dur_fw) / truth_fw
    err_bw = (truth_bw - dur_bw) / dur_bw
    err_fw_list.append(err_fw)
    err_bw_list.append(err_bw)
    print(f"[{idx}/{len(data)}]\t {np.mean(fw_measure) : .3f}, {np.mean(bw_measure) : .3f}, {dur_fw : .3f}, {dur_bw : .3f},")

print("forward error:", np.mean(err_fw_list))
print("backward errpr:", np.mean(err_bw_list))
# prof.export_chrome_trace("trace_tc.json")