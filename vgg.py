import torch
import time
from torch import nn
import numpy as np
from torch.autograd import DeviceType

import argparse
torch.backends.cudnn.benchmark = True

def build_vgg_model(bias=True):
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
    layers.append(nn.LazyLinear(4096, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.LazyLinear(1000, bias=bias))
    return nn.Sequential(*layers)


if __name__ == '__main__':
    record = []


    image_size = 224
    # print("batch, pred, truth, error")
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    model = build_vgg_model(False)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    target_batch_size = 4
    for _ in range(1, 1000):
        record = []
        def analyze(prof):
            events = prof.profiler.function_events
            for evt in events:
                if evt.name.startswith("ProfilerStep"):
                    record.append(evt.cpu_time_total)
                if evt.name.startswith("cudaDeviceSynchronize"):
                    record[-1] -= evt.cpu_time_total
        inputs = torch.rand((target_batch_size, 3, image_size, image_size), device=device)
        labels = torch.randint(999, (target_batch_size, ), device=device)
        model(inputs)
        x = inputs
        dur_tot = 0
        nitr = 20
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule= torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=nitr,
                repeat=0),
            with_stack=True,
            record_shapes=True,
            on_trace_ready=analyze
        ) as profiler:
            for _ in range(nitr+10):
                optim.zero_grad(set_to_none=True)
                x = inputs
                x = model(x)
                loss = loss_fn(x, labels)
                loss.backward()
                optim.step()
                torch.cuda.synchronize()
                profiler.step()
        # dur_tot = np.mean(record)
        for d in record:
            print(d / 1e3)
        # profiler.export_chrome_trace("trace.json")
        # print(dur_tot / 1e3)