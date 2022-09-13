import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import timing

from predictor import Conv2DPredictor

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument('command', choices=['measure', 'plot'])
parser.add_argument('--use_fp16', action='store_true')
parser.add_argument('--pred', action='store_true')
parser.add_argument('--nomodulo', action='store_true')
args = parser.parse_args() 

log_path = f"./data/gpu_noncontinuous_{args.use_fp16}_{args.pred}_{args.nomodulo}.npz"


if args.pred:
    conv_pred = Conv2DPredictor(not args.nomodulo)
    conv_pred.load_model(f"./model/predictor_model_conv2d{'_nomodulo' if args.nomodulo else ''}.th")

def dtype(args):
    if args.use_fp16:
        return torch.float16
    else:
        return torch.float32

def change_one_dim(
    default,
    param_name,
    param_range
):
    param_list = ['batch_size', 'image_size', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
    idx = param_list.index(param_name)
    params = default.copy()
    dur_list = []
    pred_list = []
    for v in tqdm(param_range):
        params[idx] = v
        batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding = params
        layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device) 
        x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, dtype=dtype(args))
        if args.use_fp16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                dur = timing(lambda : layer(x))
        else:
            dur = timing(lambda : layer(x)) 
        if args.pred:
            pred_list.append(
                conv_pred.predict(
                    [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, args.use_fp16]
                )
            )
        else:
            pred_list.append(1)
        dur_list.append(dur)
    return np.array([param_range, dur_list, pred_list])

if args.command == 'measure':
    default = [16, 224, 64, 64, 3, 1, 1]
    bs_data = change_one_dim(
        default,
        'batch_size',
        range(1, 65),
    )

    oc_data = change_one_dim(
        default,
        'out_channels',
        range(4, 515, 4),
    )

    ic_data = change_one_dim(
        default,
        'in_channels',
        range(4, 515, 4),
    )


    np.savez(log_path, change_batch_size=bs_data, change_out_channels=oc_data, change_in_channels=ic_data)
elif args.command == 'plot':
    data = np.load(log_path)
    bs_data = data['change_batch_size']
    oc_data = data['change_out_channels']
    ic_data = data['change_in_channels']

    plt.figure()
    plt.subplot(311)
    plt.xlabel("batch size")
    plt.ylabel("time (ms)")
    plt.plot(bs_data[0], bs_data[1])
    if args.pred:
        plt.plot(bs_data[0], bs_data[2])

    plt.subplot(312)
    plt.xlabel("number of output channels")
    plt.ylabel("time (ms)")
    plt.plot(oc_data[0], oc_data[1])
    if args.pred:
        plt.plot(oc_data[0], oc_data[2])

    plt.subplot(313)
    plt.xlabel("number of in channels")
    plt.ylabel("time (ms)")
    plt.plot(ic_data[0], ic_data[1])
    if args.pred:
        plt.plot(ic_data[0], ic_data[2])
    
    plt.subplots_adjust(hspace=0.6)
 
    plt.savefig(f"./figure/gpu_noncontinuous{'_fp16' if args.use_fp16 else ''}{'_nomodulo' if args.nomodulo else ''}{'_pred' if args.pred else ''}.png")
