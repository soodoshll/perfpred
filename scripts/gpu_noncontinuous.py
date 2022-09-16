import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from tqdm import tqdm, trange

from perfpred.utils import timing

from perfpred.predictor import Conv2DPredictor

torch.backends.cudnn.benchmark = True

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
    default = [32, 224, 64, 64, 3, 1, 1]
    batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding = default
    print("warm up")
    warmup = 10_000
    x = torch.rand(batch_size, in_channels, image_size, image_size, device=device)
    model = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=device)
    for _ in trange(warmup):
        model(x)
    torch.cuda.synchronize()

    bs_data = change_one_dim(
        default,
        'batch_size',
        range(1, 65),
    )

    oc_data = change_one_dim(
        default,
        'out_channels',
        range(4, 385, 4),
    )

    ic_data = change_one_dim(
        default,
        'in_channels',
        range(4, 385, 4),
    )

    np.savez(log_path, change_batch_size=bs_data, change_out_channels=oc_data, change_in_channels=ic_data)

elif args.command == 'plot':
    data = np.load(log_path)
    bs_data = data['change_batch_size']
    oc_data = data['change_out_channels']
    ic_data = data['change_in_channels']

    def draw(ax, data, name):
        ax.set_xlabel(name)
        ax.set_ylabel("time (ms)")
        ax.plot(data[0], data[1], label='truth')
        if args.pred:
            ax.plot(data[0], data[2], label='pred')
        ax2 = ax.twinx()
        err = (data[2] - data[1]) / data[1]
        ax2.plot(data[0], err, 'k--', label="error")
        ax2.grid()
        # vals = ax2.get_yticks()
        # ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=[8, 8])
    draw(axes[0], bs_data, "batch size")
    draw(axes[1], oc_data, "num of output channels")
    draw(axes[2], ic_data, "num of input channels")

    # plt.subplot(312)
    # plt.xlabel("number of output channels")
    # plt.ylabel("time (ms)")
    # plt.plot(oc_data[0], oc_data[1], label='truth')
    # if args.pred:
    #     plt.plot(oc_data[0], oc_data[2], label='pred')

    # plt.subplot(313)
    # plt.xlabel("number of in channels")
    # plt.ylabel("time (ms)")
    # plt.plot(ic_data[0], ic_data[1], label='truth')
    # if args.pred:
    #     plt.plot(ic_data[0], ic_data[2], label='pred')
    
    plt.subplots_adjust(hspace=0.6)
    axes[0].legend()
 
    plt.savefig(f"./figure/gpu_noncontinuous{'_fp16' if args.use_fp16 else ''}{'_nomodulo' if args.nomodulo else ''}{'_pred' if args.pred else ''}.png")
