import torch
import torchvision
import torch.nn.functional as F
import time
import numpy as np
from perfpred.utils import timing
from perfpred import trace
device = 'cuda'
from matplotlib import pyplot as plt


# model = ResBlock(64, 64)
nitr = 20
model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
# model_names = ["resnet18"]
time_before =[]
time_after = []
pred = []

for model_name in model_names:
    bs = 32
    x = torch.rand(bs, 3, 224, 224, device=device)
    model = getattr(torchvision.models, model_name)()
    model.to(device)
    model.eval()

    for _ in range(3):
        model(x)
    dur = timing(lambda : model(x), nitr)
    print("before compiled:", dur)

    traced_model = torch.jit.trace(model, (x, ))
    print("tracing completes")

    for _ in range(3):
        traced_model(x)
    print("start profiling compiled version")
    dur1 = timing(lambda : traced_model(x), nitr)
    print("after compiled:", dur1)

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

    p = io_size * trace.UNARY_COEFF / 2 + trace.UNARY_BIAS * (len(g_list) - 1)
    time_before.append(dur)
    time_after.append(dur1)
    pred.append(p)

    del model, traced_model, graph, groups, group_nodes
    torch.cuda.empty_cache()



time_before = np.array(time_before)
time_after = np.array(time_after)

labels = model_names
time_before_minus_pred = time_before - pred

width = 0.35
x = np.arange(len(labels))

plt.bar(x - width/2, time_before_minus_pred, width=width, edgecolor="black")
rects1 = plt.bar(x - width/2, pred, bottom=time_before_minus_pred, width=width, label="predicted acceleration", color="cyan", edgecolor="black")
rects2 = plt.bar(x + width/2, time_after, width=width, label="running time after fusion", color="orange", edgecolor="black")
plt.xticks(x, labels)
plt.ylabel("Time (ms)")
plt.legend()

# plt.bar_label(rects1, padding=3)
# plt.bar_label(rects2, padding=3)
plt.savefig('nvfuser_resnets.png')
