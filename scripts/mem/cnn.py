import torch
from torch import nn
import torchvision
import argparse
import os, subprocess
from perfpred.utils import measure_gpu_mem
from torch.cuda.amp import GradScaler
import time

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--nitr', type=int, default=10)
parser.add_argument('--amp', action="store_true")

args = parser.parse_args()
if 'LD_PRELOAD' in os.environ:
    use_fake_alloc = True
    import fake_alloc
    CNN_COMPENSATE = 1690 * 1024 * 1024
    fake_alloc.set_target_mem_limit(24 * 1024 * 1024 * 1024 - CNN_COMPENSATE)
    # torch.set_target_mem_limit(24 * 1024 * 1024 * 1024 - CNN_COMPENSATE)
else:
    use_fake_alloc = False

device = 'cuda'
model = getattr(torchvision.models, args.model)()
device = torch.device('cuda')
model.to(device)
model.train()

optim = torch.optim.SGD(model.parameters(), lr=1e-3)
x = torch.empty((args.batch_size, 3, args.image_size, args.image_size), device=device)
label = torch.zeros((args.batch_size, ), dtype=torch.int64, device=device)
loss_fn = nn.CrossEntropyLoss()


scaler = GradScaler(enabled=args.amp)

def train(nitr=2):
    for _ in range(nitr):
        optim.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
            out = model(x)
            if args.model == 'inception_v3':
                out = out.logits
            loss = loss_fn(out, label)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    torch.cuda.synchronize()

# warm up
train()
torch.cuda.reset_peak_memory_stats()
time.sleep(1)
if use_fake_alloc:
    train(args.nitr)
    # print((fake_alloc.max_mem_allocated() + CNN_COMPENSATE) / (1024)**2)
    print((torch.cuda.max_memory_reserved() + CNN_COMPENSATE) / (1024)**2)
else:
    max_mem = measure_gpu_mem(lambda: train(args.nitr))
    print(max_mem)
