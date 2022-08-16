from vgg import build_vgg_model
import torch
from torch import nn
from contextlib import suppress
import os
import argparse
import time
from apex import amp
import torchvision
# from functorch import make_functional
# from functorch.compile import aot_module, min_cut_rematerialization_partition, nop, memory_efficient_fusion
import faulthandler
faulthandler.enable()

parser = argparse.ArgumentParser()
parser.add_argument('batch_size', type=int)
parser.add_argument('--amp', type=int, default=0)
parser.add_argument('--model', type=str, default="vgg16")
args = parser.parse_args()

use_fake_alloc = os.environ.get("LD_PRELOAD", None) == "./fake_libcudart.so"
print("Using fake allocator:", use_fake_alloc)
if use_fake_alloc:
    import fake_alloc
    fake_alloc.set_target_mem_limit(16_000_000_000)
# 
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# print(torch.backends.cudnn.version())
# torch.backends.cudnn.allow_tf32 = False
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.enabled = False

# model = build_vgg_model()
# class SavedTensorRecord(object):
#     saved_tensor_size = 0

#     @property
#     def pack_hook(self):
#         def foo(x):
#             self.saved_tensor_size += x.element_size() * x.nelement()
#             return x
#         return foo

#     @property
#     def unpack_hook(self):
#         def foo(x):
#             return x
#         return foo
    
#     def reset(self):
#         self.saved_tensor_size = 0

# saved_tensor_record = SavedTensorRecord()

if args.model == 'vgg':
    model = build_vgg_model(False)
else:
    model = getattr(torchvision.models, args.model)()
device = torch.device('cuda')
model.to(device)
print("model created")

parameter_size = 0
for p in model.parameters():
    parameter_size += p.data.element_size() * p.data.nelement()

print("parameters: ", parameter_size)

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
    
t = torch.randint(1000, (bs, ), device=device)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

# model, optim = amp.initialize(model, optim, opt_level="O" + str(args.amp))


for i in range(5):
    optim.zero_grad(set_to_none=True)
    # saved_tensor_record.reset()
    if i == 1:
        torch.cuda.synchronize()
        print("No benchmark anymore!")
    if i == 4:
        torch.cuda.synchronize()
        if use_fake_alloc:
            fake_alloc.init_max_mem()
        else:
            torch.cuda.reset_peak_memory_stats()
    # print("run forward")
    # with torch.no_grad():
    # with torch.autograd.graph.saved_tensors_hooks(saved_tensor_record.pack_hook, saved_tensor_record.unpack_hook):
    out = model(x)
    # loss = loss_fn(out, t)
    loss = out.sum()
    # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
    # print("run backward")
    loss.backward()
    optim.step()
        # del out,loss
    # torch.cuda.empty_cache()

del x, t, model
torch.cuda.synchronize()
# print(saved_tensor_record.saved_tensor_size)
if use_fake_alloc:
    print(fake_alloc.max_mem_allocated())
else:
    print(torch.cuda.max_memory_allocated())
