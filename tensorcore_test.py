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
in_channels = 64
image_size = 224
warm_up = 5
nitr = 20

out_channels = 64
model = nn.Conv2d(in_channels, out_channels, 3, bias=False)
# model = torchvision.models.resnet18()
model.to(device)
x = torch.rand((batch_size, in_channels, image_size, image_size), device=device)
model(x)
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

# model, optim = amp.initialize(model, optim, opt_level="O3")
scaler = torch.cuda.amp.GradScaler()

for batch_size in range(32, 32 + 1, 1):
    x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, requires_grad=True, dtype=torch.float16)
    def foo():
        optim.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            out = model(x)
            loss = out.sum()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
            # scaled_loss.backward()
        # optim.step()
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    
    for _ in range(warm_up):
        foo()
    torch.cuda.synchronize()

    # dur = timing(lambda: model(x), nitr)
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    dur_tc = timing(foo, nitr)
    # print(batch_size, dur, dur_tc)
    print(batch_size, dur_tc)