import torch
import time
from torch import nn
from predictor import *
import numpy as np
from torch.autograd import DeviceType

import argparse
torch.backends.cudnn.benchmark = True

def build_vgg_model(bias):
    layers = []
    layers.append(nn.Conv2d(3, 64, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(64, 64, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Conv2d(64, 128, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(128, 128, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Conv2d(128, 256, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(256, 256, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(256, 256, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Conv2d(256, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.Flatten())
    layers.append(nn.LazyLinear(4096, bias=bias))
    layers.append(nn.ReLU())
    layers.append(nn.LazyLinear(1000, bias=bias))
    return nn.Sequential(*layers)


if __name__ == '__main__':
    verbose = 0
    modulo = True
    model_name = "predictor_model_conv2d.th" if modulo else "predictor_model_conv2d_0.th"

    # linear_pred = LinearPredictor()
    # linear_pred.load_model("predictor_model_linear.th")
    # # linear_pred.test_auto()
    # linear_pred.model.to(torch.device('cpu'))

    conv_pred = Conv2DPredictor(modulo)
    # conv_pred.load_data(CONV2D_PATH)
    conv_pred.load_model(model_name)
    conv_pred.model.to(torch.device('cpu'))
    # conv_pred.xgb_fit()

    # maxpool_pred = MaxPoolingPredictor()
    # maxpool_pred.load_model("predictor_model_maxpool.th")
    # maxpool_pred.model.to(torch.device('cpu'))

    record = []
    record_bw = []
    def analyze(prof):
        events = prof.profiler.function_events
        for evt in events:
            if evt.device_type == DeviceType.CPU:
                # print(evt.name, evt.kernels)
                duration = sum([k.duration for k in evt.kernels]) / 1e3
                # if evt.name in ["aten::cudnn_convolution", "aten::mm", "aten::max_pool2d_with_indices", "aten::relu"]:
                if evt.name == "aten::cudnn_convolution":
                    record.append(duration)
                if evt.name == "aten::convolution_backward":
                    record_bw.append(duration)

    image_size = 128
    # print("batch, pred, truth, error")
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # for target_batch_size in range(1, 60):
    # target_batch_size = 41
    image_size_list = []
    run_time_list = []
    for target_batch_size in range(1, 65):
        record = []
        record_bw = []
        model = build_vgg_model(False)
        model.to(device)
        inputs = torch.rand((target_batch_size, 3, image_size, image_size), device=device)
        model(inputs)
        x = inputs
        dur_tot = 0
        nitr = 10
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule= torch.profiler.schedule(
                wait=1,
                warmup=20,
                active=nitr,
                repeat=0),
            with_stack=True,
            record_shapes=True,
            on_trace_ready=analyze
        ) as profiler:
            for _ in range(nitr+21):
                x = inputs
                x = model(x)
                x = x.sum()
                x.backward()
                profiler.step()
            torch.cuda.synchronize()
        dur_tot = sum(record) / nitr

        record_bw.reverse()

        x = inputs
        pred_tot = 0
        n = 0
        pred_fw = []
        pred_bw = []
        for layer in model:
            if isinstance(layer, nn.Conv2d):
                n += 1
                pred_forward = conv_pred.predict(
                    [0, 
                    target_batch_size, 
                    x.shape[2], 
                    x.shape[1], 
                    layer.out_channels, 
                    layer.kernel_size[0], 
                    layer.stride[0], 
                    layer.padding[0], 
                    1]
                )
                pred_backward = conv_pred.predict(
                    [0, 
                    target_batch_size, 
                    x.shape[2], 
                    x.shape[1], 
                    layer.out_channels, 
                    layer.kernel_size[0], 
                    layer.stride[0], 
                    layer.padding[0], 
                    0]
                )
                if n==1:
                    pred_backward /= 2
                pred_fw.append(pred_forward)
                pred_bw.append(pred_backward)
                # pred_tot += pred
                # print(pred_forward, pred)
            # elif isinstance(layer, nn.Linear):
            #     pred = linear_pred.predict(
            #         [0,
            #         target_batch_size,
            #         x.shape[1],
            #         layer.out_features,
            #         1
            #         ]
            #     )
            #     pred_tot += pred
            # elif isinstance(layer, nn.MaxPool2d):
            #     pred = maxpool_pred.predict(
            #         [target_batch_size,
            #          layer.kernel_size,
            #          x.shape[2],
            #          x.shape[1],
            #          layer.stride,
            #          1
            #         ]
            #     )
            #     pred_tot += pred
            x = layer(x)

        if verbose >= 2:
            print(f"===== batch size: {target_batch_size} =====")
            print(len(record), len(pred_fw))
            for fw, bw, pfw, pdw in zip(record, record_bw, pred_fw, pred_bw):
                print(fw, pfw, bw, pdw)
            print(sum(record), sum(pred_fw), sum(record_bw), sum(pred_bw))
        fw_truth = sum(record) / nitr
        bw_truth = sum(record_bw) / nitr
        fw_pred = sum(pred_fw)
        bw_pred = sum(pred_bw)
        dur_truth = fw_truth + bw_truth
        dur_pred = sum(pred_fw) + sum(pred_bw)

        image_size_list.append(image_size)
        run_time_list.append(fw_truth)    
        if verbose == 0:
            print(f"{target_batch_size}, {target_batch_size / dur_truth : .3f}, {target_batch_size / dur_pred :.3f}")
        elif verbose == 1:
        # print(target_batch_size, pred_tot, dur_tot, len(record) / 10, n)
            print(f"{target_batch_size}, {fw_truth : .2f}, {fw_pred : .2f}, {bw_truth : .2f}, {bw_pred :.2f}")
        # print(f"{target_batch_size}, {target_batch_size / dur_pred}, {target_batch_size / dur_truth}, {abs(dur_pred - dur_truth) / dur_truth * 100: .2f}% ")

    plt.plot(image_size_list, run_time_list)
    plt.savefig("perf_image_size.png")