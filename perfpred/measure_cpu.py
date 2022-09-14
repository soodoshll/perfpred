import torch
from torch import nn
import numpy as np

from measure import measure_op, ConvMeasure, MatMulMeasure, BatchNormMeasure
from torch.autograd import DeviceType
from matplotlib import pyplot as plt
import argparse, pickle

parser = argparse.ArgumentParser()
parser.add_argument("op", choices=["conv2d", "mm", "batchnorm", "all"])
parser.add_argument("command", choices=["measure", "plot", "measure_time"])
args = parser.parse_args()

def _find_next(events, last):
    for evt in events:
        if evt.time_range.start >= last.time_range.end:
            return evt

class ConvMeasureCPU(ConvMeasure):
    def get_measured_func(self):
        def func(data):
            A = data[0]
            layer = data[1]
            layer.train()
            out = layer(A)
            torch.cuda.synchronize(self.device)
        return func

    def get_analyze_func(self):
        # print("Using CPU events analyzing function")
        def func(info, prof):
            events = prof.profiler.function_events
            real_durs = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                # if evt.device_type == DeviceType.CPU and evt.cpu_parent is None:
                if evt.name == 'aten::conv2d':
                    dur = evt.cpu_time_total
                    real_dur = _find_next(events, evt).time_range.start - evt.time_range.start
                    real_durs.append(real_dur)
                    info['data'].append(real_dur)
        return func

class MatMulMeasureCPU(MatMulMeasure):
    def get_measured_func(self):
        def func(data):
            A = data[0]
            B = data[1]
            C = A.matmul(B)
        return func

    def get_analyze_func(self):
        # print("Using CPU events analyzing function")
        def func(info, prof):
            events = prof.profiler.function_events
            real_durs = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                # if evt.device_type == DeviceType.CPU and evt.cpu_parent is None:
                if evt.name == 'aten::mm':
                    dur = evt.cpu_time_total
                    real_dur = _find_next(events, evt).time_range.start - evt.time_range.start
                    # print(self.params, "#", real_dur)
                    real_durs.append(real_dur)
                    info['data'].append(real_dur)
        return func    

class BatchNormMeasureCPU(BatchNormMeasure):
    def get_measured_func(self):
        def func(data):
            A = data[0]
            layer = data[1]
            layer.train()
            out = layer(A)
            # out = out.sum()
            # out.backward()
            # torch.cuda.synchronize(self.device)
        return func

    def get_analyze_func(self):
        # print("Using CPU events analyzing function")
        def func(info, prof):
            events = prof.profiler.function_events
            real_durs = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                # if evt.device_type == DeviceType.CPU and evt.cpu_parent is None:
                if evt.name == 'aten::batch_norm':
                    dur = evt.cpu_time_total
                    real_dur = _find_next(events, evt).time_range.start - evt.time_range.start
                    # print(self.params, "#", real_dur)
                    real_durs.append(real_dur)
                    info['data'].append(real_dur)
        return func 

def read_data(filename):
    durs = []
    with open(filename, "rb") as f:
        while 1:
            try:
                objs = pickle.load(f)
                for obj in objs:
                    durs.append(obj / 1000)
            except (EOFError):
                break
    durs = np.array(durs)
    return durs

CONV_FILENAME= "data/conv_data_cpu.data"
MM_FILENAME = "data/mm_data_cpu.data"
BATCHNORM_FILENAME="data/batchnorm_cpu.data"

fn_dict = {
    "conv2d": CONV_FILENAME,
    "mm": MM_FILENAME,
    "batchnorm": BATCHNORM_FILENAME
}

if args.command == "measure":
    if args.op == "conv2d":
        conv_measure_cpu = ConvMeasureCPU(
            batch_size_range=(1, 48),
            image_size_range=(2, 224),
            in_channels_range=(3, 1024),
            out_channels_range=(16, 1024),
            kernel_size_range=(1, 7),
            stride_range=(1, 7),
            padding_range=(1, 3),
            device=torch.device('cuda:0')
        )
        conv_measure_cpu.run(1000, dx=True, filename=CONV_FILENAME)
    elif args.op == "mm":
        matmul_measure_cpu = MatMulMeasureCPU(
            n_range=(1, 1024),
            m_range=(1, 16384),
            k_range=(1, 16384),
            device=torch.device('cuda:0')
        )
        matmul_measure_cpu.run(1000, filename=MM_FILENAME)
    elif args.op == "batchnorm":
        batchnorm_measure_cpu = BatchNormMeasureCPU(
            batch_size_range=(1, 64),
            image_size_range=(2, 224),
            channels_range=(1, 1024),
            device=torch.device(f'cuda:0')
        )
        batchnorm_measure_cpu.run(1000, filename=BATCHNORM_FILENAME)
    else:
        raise RuntimeError("Not Supported")
elif args.command == "plot":
    if args.op == "all":
        conv_data = read_data(CONV_FILENAME)
        mm_data = read_data(MM_FILENAME)
        bn_data = read_data(BATCHNORM_FILENAME)
        labels = ["conv2d", "matmul", "batch_norm"]

        plt.violinplot([conv_data, mm_data, bn_data], showmeans=True, showextrema=False, showmedians=True)
        plt.xticks(np.arange(1, len(labels) + 1), labels=labels)
        plt.xlim(0.25, len(labels) + 0.75)
        plt.ylim(0, 2)
        plt.ylabel('time (ns)')
        plt.savefig("figure/cpu_overhead.png")
    else:
        data = read_data(fn_dict[args.op])
        plt.violinplot(data, showextrema=False, showmeans=True)
        plt.savefig("figure/cpu_overhead_e2e.png")