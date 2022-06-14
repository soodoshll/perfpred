import torch
import time
from torch import nn
from dataloading import *
import numpy as np
from torch.autograd import DeviceType

def build_vgg_model(bias):
    layers = []
    layers.append(nn.Conv2d(3, 64, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(64, 64, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Conv2d(64, 128, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(128, 128, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Conv2d(128, 256, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(256, 256, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(256, 256, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Conv2d(256, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Flatten())
    layers.append(nn.LazyLinear(4096))
    layers.append(nn.ReLU())
    layers.append(nn.LazyLinear(1000))
    return nn.Sequential(*layers)

model = build_vgg_model(True)
device = torch.device('cuda:2')
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True
model.to(device)
conv_pred = Conv2DPredictor()
# conv_pred.load_data(CONV2D_PATH)
conv_pred.load_model("predictor_model_conv2d.th")
conv_pred.model.to(torch.device('cpu'))
# conv_pred.xgb_fit()

record = []
def analyze(prof):
    events = prof.profiler.function_events
    for evt in events:
        if evt.device_type == DeviceType.CPU:
            duration = sum([k.duration for k in evt.kernels]) / 1e3
            if evt.name == "aten::cudnn_convolution":
                # print(duration)
                record.append(duration)

image_size = 224
print("batch, pred, truth, error")
for target_batch_size in range(4, 65):
    record = []
    inputs = torch.rand((target_batch_size, 3, image_size, image_size), device=device)
    model(inputs)
    x = inputs
    dur_tot = 0
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        on_trace_ready=analyze
    ) as profiler:
        for _ in range(10):
            x = inputs
            for layer in model:
                x = layer(x)
        torch.cuda.synchronize()

    x = inputs
    pred_tot = 0
    for layer in model:
        if isinstance(layer, nn.Conv2d):
            pred = conv_pred.predict(
                [0, 
                target_batch_size, 
                x.shape[2], 
                x.shape[1], 
                layer.out_channels, 
                layer.kernel_size[0], 
                layer.stride[0], 
                layer.padding[0], 
                1]
            )
            # print(pred)
            pred_tot += pred
        x = layer(x)

    dur_tot = sum(record) / 10
    print(f"{target_batch_size}, {pred_tot}, {dur_tot}, {abs(pred_tot - dur_tot) / dur_tot * 100: .2f}% ")