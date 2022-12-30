import torch
from torch import nn
import numpy as np
import time
import matplotlib
from matplotlib import pyplot as plt
import torch.nn.functional as F
from perfpred.measure import measure_unary_elementwise, measure_binary_elementwise

x_range = np.arange(200_000, 11_000_000, 200_000)
unary_ops = [F.sigmoid, F.relu, F.tanh, lambda x: x+1, F.gelu]
unary_captions = ['sigmoid', 'relu', 'tanh', '+1', 'gelu', ]

unary_ys = {cap:[] for cap in unary_captions}
for op, caption in zip(unary_ops, unary_captions):
    for n in x_range:
        dur = measure_unary_elementwise(n, op=op)
        unary_ys[caption].append(dur)
for cap in unary_captions:
    plt.plot(x_range, unary_ys[cap], label=cap)

# unary_y_pred = 2 * 4 * x_range / (616e9) * 1e3
# plt.plot(x_range, unary_y_pred, label='pred_unary')

ret, _, _, _ = np.linalg.lstsq(
    np.expand_dims(x_range, axis=1), 
    np.array(unary_ys['relu']) )
print(ret, 2*4 / (616e9) * 1e3)
print(f"{1/ret  * 4 * 2 / 1e6} GB/s")

binary_ops = [
    torch.add,
    torch.mul,
    torch.sub,
    torch.max,
    # torch.eq,
    # # torch.greater
]

binary_captions = ['add', 'mul', 'sub', 'max']
binary_ys = {cap:[] for cap in binary_captions}
for op, caption in zip(binary_ops, binary_captions):
    for n in x_range:
        dur = measure_binary_elementwise(n, op=op)
        binary_ys[caption].append(dur)
for cap in binary_captions:
    plt.plot(x_range, binary_ys[cap], '--', label=cap)

# binary_y_pred = 3 * 4 * x_range / (616e9) * 1e3
# plt.plot(x_range, binary_y_pred, '--', label='pred_binary')
ret, _, _, _ = np.linalg.lstsq(
    np.expand_dims(x_range, axis=1), 
    np.array(binary_ys['add']) )
print(ret, 3*4 / (616e9) * 1e3)
print(f"{1/ret  * 4 * 3 / 1e6} GB/s")

matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rcParams['font.size'] = 7

plt.legend()
plt.xlabel('Number of Elements')
plt.ylabel('Time (ms)')
plt.savefig('figure/simple_op.pdf')