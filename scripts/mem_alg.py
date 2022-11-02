import torch
import torchvision
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', action='store_true')
parser.add_argument('--deterministic', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = args.benchmark
torch.backends.cudnn.deterministic = args.deterministic

model = torchvision.models.vgg11()
model.to('cuda')
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

x = torch.rand(64, 3, 224, 224, device='cuda')

def train_loop():
    optim.zero_grad(set_to_none=True)
    out = model(x)
    out.sum().backward()
    optim.step()

train_loop()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
train_loop()
torch.cuda.synchronize()
max_mem_allocated = torch.cuda.max_memory_reserved()


# torch.backends.cudnn.benchmark = True
# train_loop()
# torch.cuda.reset_peak_memory_stats()
# train_loop()
# torch.cuda.synchronize()
# max_mem_allocated_benchmark = torch.cuda.max_memory_allocated()

print(max_mem_allocated)