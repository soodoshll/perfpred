import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import timing

log_path = "./data/gpu_noncontinuous.npz"
device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument('command', choices=['measure', 'plot'])
args = parser.parse_args() 

def change_batch_size(
    batch_size_list,
    image_size, in_channels, out_channels, kernel_size, stride, padding
):
    layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    dur_list = []
    for batch_size in tqdm(batch_size_list):
        x = torch.rand((batch_size, in_channels, image_size, image_size), device=device)
        dur = timing(lambda : layer(x))
        dur_list.append(dur)
    return np.array([batch_size_list, dur_list])

def change_out_channels(
    out_channel_list,
    batch_size, image_size, in_channels, kernel_size, stride, padding
):
    dur_list = []
    for out_channels in tqdm(out_channel_list):
        layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
        x = torch.rand((batch_size, in_channels, image_size, image_size), device=device)
        dur = timing(lambda : layer(x))
        dur_list.append(dur)
    return np.array([out_channel_list, dur_list])   

if args.command == 'measure':
    bs_data = change_batch_size(
        range(1, 65),
        224, 64, 64, 3, 1, 1
    )

    oc_data = change_out_channels(
        range(4, 515, 4),
        32, 224, 64, 3, 1, 1
    )

    np.savez(log_path, change_batch_size=bs_data, change_out_channels=oc_data)
elif args.command == 'plot':
    data = np.load(log_path)
    bs_data = data['change_batch_size']
    oc_data = data['change_out_channels']

    plt.figure()
    plt.subplot(211)
    plt.xlabel("batch size")
    plt.ylabel("time (ms)")
    plt.plot(bs_data[0], bs_data[1])

    plt.subplot(212)
    plt.xlabel("number of output channels")
    plt.ylabel("time (ms)")
    plt.plot(oc_data[0], oc_data[1])
    plt.subplots_adjust(hspace=0.4)
 
    plt.savefig("./figure/gpu_noncontinuous.png")
