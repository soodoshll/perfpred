from vgg import build_vgg_model
import torch
from torch import nn
from contextlib import suppress
import os
import argparse
import time
from apex import amp
# from functorch import make_functional
# from functorch.compile import aot_module, min_cut_rematerialization_partition, nop, memory_efficient_fusion

parser = argparse.ArgumentParser()
parser.add_argument('batch_size', type=int)
parser.add_argument('--amp', type=int, default=0)
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

model = build_vgg_model()
device = torch.device('cuda')
model.to(device)
print("model created")

# while True:
#     pass
# func_model, params = make_functional(model)
# compiled_model = memory_efficient_fusion(model)

def who_cares_loss(out, label):
    return torch.sum(out)

bs = args.batch_size
image_size = 224
x = torch.rand((bs, 3, image_size, image_size), device=device)
model(x)
if use_fake_alloc:
    fake_alloc.init_max_mem()
else:
    torch.cuda.reset_max_memory_allocated()
    
# t = torch.randint(1000, (bs, ), device=device)
loss_fn = who_cares_loss
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optim = amp.initialize(model, optim, opt_level="O" + str(args.amp))

optim.zero_grad()
print("run forward")
out = model(x)
# out = compiled_model(x)
# loss = loss_fn(out, t)
loss = torch.sum(out)
with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
    loss.backward()
optim.step()

if use_fake_alloc:
    print(fake_alloc.max_mem_allocated())
else:
    print(torch.cuda.max_memory_allocated())
