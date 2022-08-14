from vgg import build_vgg_model
import torch
from torch import nn
from contextlib import suppress
import os
import argparse
from apex import amp
import time
import torchvision
# from functorch import make_functional
# from functorch.compile import aot_module, min_cut_rematerialization_partition, nop, memory_efficient_fusion

parser = argparse.ArgumentParser()
parser.add_argument('batch_size', type=int)
parser.add_argument('--amp', type=int, default=0)
parser.add_argument('--nitr', type=int, default=10)
args = parser.parse_args()

use_fake_alloc = os.environ.get("LD_PRELOAD", None) == "./fake_libcudart.so"
print("Using fake allocator:", use_fake_alloc)
if use_fake_alloc:
    import fake_alloc

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# print(torch.backends.cudnn.version())
# torch.backends.cudnn.allow_tf32 = False
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.enabled = False

# model = build_vgg_model()
model = torchvision.models.resnet18()
device = torch.device('cuda')
model.to(device)
print("model created")

def who_cares_loss(out, label):
    return torch.sum(out)

def reset_mem():
    if use_fake_alloc:
        fake_alloc.init_max_mem()
    else:
        torch.cuda.reset_peak_memory_stats()

bs = args.batch_size
image_size = 224
x = torch.rand((bs, 3, image_size, image_size), device=device)
model(x)
    
loss_fn = who_cares_loss
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optim = amp.initialize(model, optim, opt_level="O" + str(args.amp))
for i in range(10):
    if i == 8:
        torch.cuda.synchronize()
        t0 = time.time()
        reset_mem()
    optim.zero_grad()
    out = model(x)
    loss = torch.sum(out)
    with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
        scaled_loss.backward()
    optim.step()
    del loss, out

torch.cuda.synchronize()
dur = (time.time() - t0) / 2
print(dur)
if use_fake_alloc:
    print(fake_alloc.max_mem_allocated())
else:
    print(torch.cuda.max_memory_allocated())
