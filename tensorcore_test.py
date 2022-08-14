import torch
from torch import nn 
import time
from vgg import build_vgg_model
from apex import amp 
import torchvision

def timing(func, nitr=1):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nitr):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / nitr

device = torch.device('cuda')
batch_size = 32
in_channels = 3
image_size = 224
warm_up = 5
nitr = 10

model = torchvision.models.resnet18()
model.to(device)
x = torch.rand((batch_size, in_channels, image_size, image_size), device=device)
model(x)
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optim = amp.initialize(model, optim, opt_level="O0")
    
for batch_size in range(1, 48 + 1, 1):
    x = torch.rand((batch_size, in_channels, image_size, image_size), device=device)
    def foo():
        optim.zero_grad()
        out = model(x)
        loss = out.sum()
        with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
            scaled_loss.backward()
        optim.step()
    for _ in range(warm_up):
        foo()
    torch.cuda.synchronize()

    # dur = timing(lambda: model(x), nitr)
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    dur_tc = timing(foo, nitr)
    # print(batch_size, dur, dur_tc)
    print(batch_size, dur_tc)