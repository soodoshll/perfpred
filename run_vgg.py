from vgg import build_vgg_model
import torch
from torch import nn
from contextlib import suppress
import os
import argparse
from functorch import make_functional
from functorch.compile import aot_module, min_cut_rematerialization_partition

parser = argparse.ArgumentParser()
parser.add_argument('batch_size', type=int)
args = parser.parse_args()


use_fake_alloc = os.environ.get("LD_PRELOAD", None) == "./fake_libcudart.so"
print("Using fake allocator:", use_fake_alloc)
if use_fake_alloc:
    import fake_alloc

torch.backends.cudnn.benchmark = False

model = build_vgg_model()
device = torch.device('cuda')
model.to(device)

# func_model, params = make_functional(model)
# compiled_model = aot_module(model, partition_fn=min_cut_rematerialization_partition)

def who_cares_loss(out, label):
    return torch.sum(out)

bs = args.batch_size
x = torch.rand((bs, 3, 224, 224), device=device)
t = torch.randint(1000, (bs, ), device=device)
loss_fn = who_cares_loss
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

optim.zero_grad(set_to_none=True)
out = model(x)
# out = compiled_model(x)
loss = loss_fn(out, t)
loss.backward()
optim.step()

if use_fake_alloc:
    print(fake_alloc.max_mem_allocated())
else:
    print(torch.cuda.max_memory_allocated())
