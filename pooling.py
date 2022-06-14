import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from measure import *
from matplotlib import pyplot as plt
# ops = [
#     (F.relu, 'relu'), 
#     (torch.sigmoid, 'sigmoid'), 
#     (lambda x : x + 1, '+1'),
#     (F.gelu, 'gelu'),
#     ]
# ys = []

# x = np.arange(1, 2_000_000, 500)
# for op in ops:
#     y = []
#     print(op[1])
#     for s in tqdm(x):
#         y.append(measure_unary_elementwise(s, op=op[0], nitr=300))
#     y = np.array(y)
#     ys.append(y)

ops = [
    (F.instance_norm, 'instance_norm'), 
    (torch.relu, 'relu')
    ]
ys = []

x = np.arange(4, 64, 2)

for op in ops:
    y = []
    print(op[1], end=" ")
    for s in tqdm(x):
        y.append(measure_unary2d(s, 56, 256, op=op[0], nitr=300))
    y = np.array(y)
    ys.append(y)

for i, y in enumerate(ys):
    plt.plot(x, y, label=ops[i][1])
plt.legend()
plt.savefig("pooling.png")