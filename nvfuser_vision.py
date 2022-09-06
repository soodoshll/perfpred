import torch
import torchvision
import torch.nn.functional as F
import time
import numpy as np
device = 'cuda'


class ResBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(output_dim)
        self.conv_bias = torch.nn.Parameter(self.conv.bias.reshape(output_dim, 1, 1))
    
    def forward(self, x):
        # o = F.conv2d(x, self.conv.weight, padding=1)
        # o = o + self.conv_bias
        o = self.conv(x)
        o = self.bn(o)
        # o = F.batch_norm(o, self.bn.running_mean, self.bn.running_var)
        o = o + x
        o = F.relu(o)
        return o

# model = ResBlock(64, 64)
x = torch.rand(32, 3, 224, 224, device=device)
model = torchvision.models.resnet50()
model.to(device)
model.eval()

for _ in range(3):
    model(x)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    model(x)
torch.cuda.synchronize()
dur = time.time() - t0
print("before compiled:", dur/20)

traced_model = torch.jit.trace(model, (x, ))
print("tracing completes")

for _ in range(3):
    traced_model(x)
print("start profiling compiled version")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    traced_model(x)
torch.cuda.synchronize()
dur = time.time() - t0

print("after compiled:", dur/20)

graph = traced_model.graph_for(x)
group_nodes = graph.findAllNodes("prim::CudaFusionGroup")
groups = []

kind_counted = [
    "aten::batch_norm",
    "aten::relu"
    "aten::add"
]

io_size = 0

for node in group_nodes:
    groups.append(node.g("Subgraph"))
for group in groups:
    g_list = list(group.nodes())
    for i, node in enumerate(g_list):
        if node.kind() in kind_counted:
            io_amount = np.prod(node.output().type().sizes())
            if i == 0 or i == len(g_list) - 1:
                io_size += io_amount
            else:
                io_size += 2 * io_amount

print(io_size)
print(io_size * 4 / 1e9 / 448)