import torch
from torch import nn
import numpy as np

from measure import measure_op, ConvMeasure, MatMulMeasure, BatchNormMeasure
from torch.autograd import DeviceType

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
            # out = out.sum()
            # out.backward()
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
                    print(self.params, "#", real_dur)
                    real_durs.append(real_dur)
            # info['data'].append()
        return func

class MatMulMeasureCPU(MatMulMeasure):
    def get_measured_func(self):
        def func(data):
            A = data[0]
            B = data[1]
            C = A.matmul(B)
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
                if evt.name == 'aten::mm':
                    dur = evt.cpu_time_total
                    real_dur = _find_next(events, evt).time_range.start - evt.time_range.start
                    print(self.params, "#", real_dur)
                    real_durs.append(real_dur)
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
                    print(self.params, "#", real_dur)
                    real_durs.append(real_dur)
        return func 

# batchnorm_measure_cpu = BatchNormMeasureCPU(
#     batch_size_range=(1, 64),
#     image_size_range=(2, 224),
#     channels_range=(1, 1024),
#     device=torch.device(f'cuda:0')
# )

# batchnorm_measure_cpu.run(1000, filename="nomatter")

# exit(0)

# matmul_measure_cpu = MatMulMeasureCPU(
#     n_range=(1, 1024),
#     m_range=(1, 16384),
#     k_range=(1, 16384),
#     device=torch.device('cuda:0')
# )

# matmul_measure_cpu.run(1000, filename='data/matmul_data_cpu.data')

# exit(0)

conv_measure_cpu = ConvMeasureCPU(
    batch_size_range=(1, 64),
    image_size_range=(2, 224),
    in_channels_range=(3, 1024),
    out_channels_range=(16, 1024),
    kernel_size_range=(1, 7),
    stride_range=(1, 7),
    padding_range=(1, 3),
    device=torch.device('cuda:0')
)

conv_measure_cpu.run(1000, dx=True, filename='data/conv_data_cpu.data')