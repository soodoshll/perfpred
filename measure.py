import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import DeviceType
import functools
from functools import partial
import numpy as np
from tqdm import tqdm, trange
import pickle

import argparse
import random
import time
import os

from multiprocessing import Process
# from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("op", choices=["conv2d", "mm", "batchnorm", "avgpool2d"])
parser.add_argument("device", choices=['2070', '2080ti', 't4', 'v100'])
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--use_fp16", action="store_true")
parser.add_argument("--data_dir", type=str, default="./data/")

args = parser.parse_args()

# device = torch.device('cuda')
torch.set_grad_enabled(True)
torch.backends.cudnn.benchmark = True
# matmul
def measure_mul(n, m, k, dry_run=5, nitr=20):
    A = torch.rand((n, m), device=device, dtype=torch.float32)
    B = torch.rand((m, k), device=device, dtype=torch.float32)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(dry_run):
        C = A.matmul(B)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(nitr):
        C = A.matmul(B)
    end_event.record()
    torch.cuda.synchronize()
    dur = start_event.elapsed_time(end_event)
    return dur

# binary element-wise>
def measure_binary_elementwise(n, device=torch.device('cuda'), op=torch.add, dry_run=10, nitr=20):
    A = torch.rand((n, ), device=device, dtype=torch.float32)
    B = torch.rand((n, ), device=device, dtype=torch.float32)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(dry_run):
        C = op(A, B)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(nitr):
        C = op(A, B)
    end_event.record()
    torch.cuda.synchronize()
    dur = start_event.elapsed_time(end_event) / nitr
    return dur

# unary element_wise
def measure_unary_elementwise(n, device=torch.device('cuda'), op=F.relu, dry_run=10, nitr=20):
    A = torch.rand((n, ), device=device, dtype=torch.float32)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(dry_run):
        C = op(A) 
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(nitr):
        C = op(A) 
    end_event.record()
    torch.cuda.synchronize()
    dur = start_event.elapsed_time(end_event) / nitr
    return dur

def measure_unary_elementwise_cpu(n, op=F.relu, dry_run=10, nitr=20):
    A = torch.rand((n, ), dtype=torch.float32)
    for _ in range(dry_run):
        C = op(A) 
    t0 = time.time()
    for _ in range(nitr):
        C = op(A) 
    dur = (time.time() - t0) / nitr
    return dur

# copy d2d
def measure_d2d(n, dry_run=10, nitr=20):
    A = torch.rand((n, ), device=device, dtype=torch.float32)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(dry_run):
        C = A.clone()
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(nitr):
        C = A.clone() 
    end_event.record()
    torch.cuda.synchronize()
    dur = start_event.elapsed_time(end_event)
    return dur

def measure_reduce(n, m, dry_run=5, nitr=20):
    A = torch.rand((n, m), device=device, dtype=torch.float32)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(dry_run):
        C = A.sum(axis=1)
    # print(C.shape)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(nitr):
        C = A.sum(axis=1)
    end_event.record()
    torch.cuda.synchronize()
    dur = start_event.elapsed_time(end_event)
    return dur

def measure_unary2d(batch_size, image_size, num_channel, op=F.instance_norm ,dry_run=5, nitr=20):
    A = torch.rand((batch_size, num_channel, image_size, image_size), device=device, dtype=torch.float32)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(dry_run):
        C = op(A)
    # print(C.shape)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(nitr):
        C = op(A)
    end_event.record()
    torch.cuda.synchronize()
    dur = start_event.elapsed_time(end_event)
    return dur

def measure_op(inputs_generator, measured_func, analyze_func, device, use_fp16=False, nitr=3):
    data = inputs_generator()
    info = {}
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        schedule= torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=nitr,
            repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        on_trace_ready=functools.partial(analyze_func, info)
    ) as profiler:
        for _ in range(nitr + 3):
            measured_func(data)
            profiler.step()
        torch.cuda.synchronize(device)
    return info

def get_children_kernel_time(event):
    kernel_time = sum([kernel.duration for kernel in event.kernels])
    for child in event.cpu_children:
        kernel_time += get_children_kernel_time(child)
    return kernel_time

def get_children_kernel_time2(event):
    print(event)
    def get_children_kernel_start_and_end(event, start_list, end_list):
        for kernel in event.kernels:
            start_list.append(kernel.time_range.start)
            end_list.append(kernel.time_range.end)
            # print(start_list, end_list)
        for child in event.cpu_children:
            get_children_kernel_start_and_end(child, start_list, end_list)
    start_list, end_list = [], []
    get_children_kernel_start_and_end(event, start_list, end_list)
    # print(start_list, end_list)
    if len(start_list) == 0 or len(end_list) == 0:
        return 0
    start = min(start_list)
    end = max(end_list)
    return end - start

class MatMulMeasure(object):
    """
    Forward and backward are the same for matmul
    """

    forward_op_kw = "aten::matmul"
    backward_op_kw = "MmBackward0"

    def __init__(self, n_range, m_range, k_range, device):
        self.n_range = n_range
        self.m_range = m_range
        self.k_range = k_range
        self.data = []
        self.record = []
        self.device = device
    
    def get_inputs_generator(self):
        n = random.randint(*self.n_range) 
        m = random.randint(*self.m_range)
        k = random.randint(*self.k_range)
        self.params = (n, m, k)
        def fun():
            A = torch.empty((n, m), device=self.device, dtype=torch.float32, requires_grad=True)
            B = torch.empty((m, k), device=self.device, dtype=torch.float32, requires_grad=True)
            return A, B
        return fun
    
    def get_measured_func(self):
        def fun(data):
            self.tmp_durations = []
            A = data[0]
            B = data[1]
            C = A.matmul(B).sum()
            C.backward()
            torch.cuda.synchronize(self.device)
        return fun

    def get_analyze_func(self):
        def fun(info, prof):
            events = prof.profiler.function_events
            tmp_forward_durations = []
            tmp_backward_durations = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                if evt.device_type == DeviceType.CPU and evt.name == self.forward_op_kw:
                    duration = get_children_kernel_time(evt) / 1e3
                    tmp_forward_durations.append(duration)
                if evt.device_type == DeviceType.CPU and evt.name == self.backward_op_kw:
                    duration = get_children_kernel_time(evt) / 1e3
                    tmp_backward_durations.append(duration)
            dur_avg_forward = np.mean(tmp_forward_durations)
            dur_avg_backward = np.mean(tmp_backward_durations)
            # print(dur_avg_forward, dur_avg_backward)
            info['data'].append((dur_avg_forward, dur_avg_backward) + self.params)      
        return fun
    
    def run(self, step=1, filename='matmul_data'):
        with open(filename, 'ab+') as f:
            for _ in range(step):
                success = False
                while not success:
                    success = True
                    try:
                        ret = measure_op(partial(self.get_inputs_generator()), self.get_measured_func(), self.get_analyze_func(), device=self.device) 
                    except RuntimeError:
                        # print("oom")
                        success = False
                pickle.dump(ret['data'], f)
                f.flush()
    
    def numpy(self):
        return np.array(self.record)

class ConvMeasure(object):
    """
    Measure forward and backward for conv2D
    """
    forward_name = "aten::cudnn_convolution"
    backward_names = ["aten::convolution_backward"]

    def __init__(self, batch_size_range, image_size_range, 
                 in_channels_range, out_channels_range, kernel_size_range, stride_range,
                 padding_range, use_fp16=True, device=torch.device('cpu')):
        self.batch_size_range = batch_size_range
        self.image_size_range = image_size_range
        self.in_channels_range = in_channels_range
        self.out_channels_range = out_channels_range
        self.kernel_size_range = kernel_size_range
        self.stride_range = stride_range
        self.padding_range = padding_range
        self.record = []
        self.use_fp16 = use_fp16

        self.device = device
        self.dx = None
    
    def get_inputs_generator(self):
        batch_size = random.randint(*self.batch_size_range)
        kernel_size = random.randint(*self.kernel_size_range)
        image_size = random.randint(max(self.image_size_range[0], kernel_size), max(self.image_size_range[1], kernel_size))
        in_channels = random.randint(*self.in_channels_range)
        out_channels = random.randint(*self.out_channels_range)
        stride = random.randint(*self.stride_range) 
        padding = random.randint(*self.padding_range)
        self.params = (batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding)
        def func(dx=True):
            self.dx = dx
            dtype = torch.float16 if self.use_fp16 else torch.float32
            A = torch.rand((batch_size, in_channels, image_size, image_size), device=self.device, requires_grad=dx, dtype=dtype)
            layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, device=self.device)
            return A, layer
        return func
    
    def get_measured_func(self):
        def func(data):
            A = data[0]
            layer = data[1]
            out = layer(A)
            out = out.sum()
            out.backward()
            torch.cuda.synchronize(self.device)

        def func_fp16(data):
            A = data[0]
            layer = data[1]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = layer(A)
                out = out.sum()
            self.scaler.scale(out).backward()
            torch.cuda.synchronize(self.device)
        return func_fp16 if self.use_fp16 else func
    
    def get_analyze_func(self):
        def func(info, prof):
            events = prof.profiler.function_events
            tmp_dur_forward = []
            tmp_dur_backward = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                if evt.device_type == DeviceType.CPU:
                    duration = get_children_kernel_time(evt) / 1e3
                    if evt.name in self.backward_names:
                        # print(evt.name)
                        tmp_dur_backward.append(duration)
                        backward_shape = evt.input_shapes
                    elif evt.name == self.forward_name:
                        tmp_dur_forward.append(duration)
                        forward_shape = evt.input_shapes
            
            dur_avg_forward = np.mean(tmp_dur_forward)
            dur_avg_backward = np.mean(tmp_dur_backward)
            info['data'].append((dur_avg_forward, dur_avg_backward, self.dx, self.use_fp16) + self.params)      
        return func

    def run(self, step=1, dx=True, use_fp16=False, filename='conv_data'):
        self.use_fp16 = use_fp16
        if use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        with open(filename, 'ab+') as f:
            for _ in trange(step):
                success = False
                while not success:
                    success = True
                    try:
                        ret = measure_op(partial(self.get_inputs_generator(), dx), self.get_measured_func(), self.get_analyze_func(), 
                        device=self.device, use_fp16 = use_fp16) 
                    except RuntimeError as e:
                        # print(e)
                        success = False
                        # torch.cuda.empty_cache()
                pickle.dump(ret['data'], f)
                f.flush()

    def numpy(self):
        return np.array(self.record_forward) #, np.array(self.record_dw), np.array(self.record_dwdx)

class BatchNormMeasure(object):
    """
    forward and backward of batchnorm2d
    """
    forward_name = "aten::batch_norm"
    backward_name = "autograd::engine::evaluate_function: CudnnBatchNormBackward0"

    def __init__(self, batch_size_range, image_size_range, channels_range, device):
        self.batch_size_range = batch_size_range
        self.image_size_range = image_size_range
        self.channels_range = channels_range

        self.record = []

        self.device = device

    def get_measured_func(self):
        def func(data):
            A = data[0]
            layer = data[1]
            layer.train()
            out = layer(A)
            out = out.sum()
            out.backward()
            torch.cuda.synchronize(self.device)
        return func

    def get_inputs_generator(self): 
        batch_size = random.randint(*self.batch_size_range)
        image_size = random.randint(*self.image_size_range)
        channels = random.randint(*self.channels_range)
        self.params = (batch_size, image_size, channels)
        def func(dx=True):
            self.dx = dx
            A = torch.rand((batch_size, channels, image_size, image_size), device=self.device, requires_grad=dx)
            layer = torch.nn.BatchNorm2d(channels, device=self.device)
            return A, layer
        return func

    def get_analyze_func(self):
        def func(info, prof):
            events = prof.profiler.function_events
            tmp_dur_forward = []
            tmp_dur_backward = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                if evt.device_type == DeviceType.CPU:
                    if evt.name == self.backward_name:
                        duration = get_children_kernel_time(evt) / 1e3
                        tmp_dur_backward.append(duration)
                        backward_shape = evt.input_shapes
                    elif evt.name == self.forward_name:
                        duration = get_children_kernel_time(evt) / 1e3
                        tmp_dur_forward.append(duration)
                        forward_shape = evt.input_shapes
            
            dur_avg_forward = np.mean(tmp_dur_forward)
            dur_avg_backward = np.mean(tmp_dur_backward)
            info['data'].append((dur_avg_forward, dur_avg_backward) + self.params)      
        return func

    def run(self, step=1, dx=True, filename='batchnorm_data'):
        with open(filename, 'ab+') as f:
            for _ in range(step):
                success = False
                while not success:
                    success = True
                    try:
                        ret = measure_op(self.get_inputs_generator(), self.get_measured_func(), self.get_analyze_func(), device=self.device) 
                    except RuntimeError:
                        # print("oom")
                        success = False
                pickle.dump(ret['data'], f)
                f.flush()
    
    def numpy(self):
        return np.array(self.record_forward), np.array(self.record_dw), np.array(self.record_dwdx)

class MaxPoolingMeasure(object):
    forward_name = "aten::max_pool2d_with_indices"
    backward_name = "aten::max_pool2d_with_indices_backward"

    def __init__(self, batch_size_range, image_size_range, channels_range,
                 kernel_size_range, stride_range, device):
        self.batch_size_range = batch_size_range
        self.image_size_range = image_size_range
        self.channels_range = channels_range
        self.kernel_size_range = kernel_size_range
        self.stride_range = stride_range

        self.record_forward = []
        self.record_backward = []
        self.device = device

    def get_inputs_generator(self): 
        batch_size = random.randint(*self.batch_size_range)
        kernel_size = random.randint(*self.kernel_size_range)
        image_size = random.randint(max(kernel_size, self.image_size_range[0]), max(kernel_size, self.image_size_range[1]))
        channels = random.randint(*self.channels_range)
        stride = random.randint(*self.stride_range)
        self.params = (batch_size, kernel_size, image_size, channels, stride)
        def func():
            A = torch.rand((batch_size, channels, image_size, image_size), device=self.device, requires_grad=True)
            layer = torch.nn.MaxPool2d(kernel_size, stride=stride)
            return A, layer
        return func
    
    def get_measured_func(self):
        def func(data):
            A = data[0]
            layer = data[1]
            out = layer(A)
            out = out.sum()
            out.backward()
            torch.cuda.synchronize(self.device)
        return func

    def get_analyze_func(self):
        def func(info, prof):
            events = prof.profiler.function_events
            tmp_dur_forward = []
            tmp_dur_backward = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                if evt.device_type == DeviceType.CPU:
                    duration = sum([k.duration for k in evt.kernels]) / 1e3
                    if evt.name == self.forward_name:
                        forward_shape = evt.input_shapes
                        tmp_dur_forward.append(duration)
                    if evt.name == self.backward_name:
                        backward_shape = evt.input_shapes
                        tmp_dur_backward.append(duration)
            dur_avg_forward = np.mean(tmp_dur_forward)
            dur_avg_backward = np.mean(tmp_dur_backward)          
            info['data'].append((dur_avg_forward, dur_avg_backward) + self.params)      
        return func 

    def run(self, step=1, filename='maxpool_data'):
        with open(filename, 'ab+') as f:
            for _ in tqdm(range(step)):
                success = False
                while not success:
                    success = True
                    try:
                        ret = measure_op(partial(self.get_inputs_generator()), self.get_measured_func(), self.get_analyze_func(), device=self.device) 
                    except RuntimeError:
                        # print("oom")
                        success = False
                pickle.dump(ret['data'], f)
                f.flush()    

    def numpy(self):
        return np.array(self.record_forward), np.array(self.record_backward)

def measure_data_collect(filename='data'):
    # matmul_measure = MatMulMeasure(
    #     n_range=(1, 1024),
    #     m_range=(1, 16384),
    #     k_range=(1, 16384)
    # )
    # print("measuring matmul")
    # matmul_measure.run(10_000)
    # matmul = matmul_measure.numpy()
    # print(matmul.shape)
    # np.savez('matmul_10000', matmul=matmul)

    # return

    conv_measure = ConvMeasure(
        batch_size_range=(1, 64),
        image_size_range=(2, 224),
        in_channels_range=(3, 1024),
        out_channels_range=(16, 1024),
        kernel_size_range=(1, 7),
        stride_range=(1, 7),
        padding_range=(1, 3),
        device=torch.device('cuda:0')
    )
    print("measuring conv")
    conv_measure.run(1000, dx=True)
    # conv_measure.run(500, dx=False)
    # # conv_fw, conv_dw, conv_dwdx = conv_measure.numpy()
    # conv_fw = conv_measure.numpy()
    # # print(conv_fw.shape, conv_dw.shape, conv_dwdx.shape)
    # # np.savez('conv_500', conv_fw=conv_fw, conv_dw=conv_dw, conv_dwdx=conv_dwdx)
    # np.savez('conv_500', conv_fw=conv_fw)
    # return 

    # bn_measure = BatchNormMeasure(
    #     batch_size_range=(1, 16),
    #     image_size_range=(2, 224),
    #     channels_range=(1, 256)
    # )
    # print("measuring bn")
    # bn_measure.run(1000, dx=True)
    # bn_measure.run(1000, dx=False)
    # bn_fw, bn_dw, bn_dwdx = bn_measure.numpy()
    # print(bn_fw.shape, bn_dw.shape, bn_dwdx.shape)

    pool_measure = MaxPoolingMeasure(
        batch_size_range=(1, 32),
        image_size_range=(2, 256),
        channels_range=(1, 1024),
        kernel_size_range=(1, 7),
        stride_range=(1, 4)
    )
    print("measure maxpooling")
    pool_measure.run(10_000)
    pool_fw, pool_bw = pool_measure.numpy()
    np.savez('pool_10000', pool_fw=pool_fw, pool_bw=pool_bw)

def mp_measure_conv(gpu_id, device_type, use_fp16=False):
    max_batch_size = 48 if device_type == "2070" else 64
    conv_measure = ConvMeasure(
        batch_size_range=(1, 64),
        image_size_range=(2, 224),
        in_channels_range=(3, 1024),
        out_channels_range=(16, 1024),
        kernel_size_range=(1, 7),
        stride_range=(1, 7),
        padding_range=(0, 3),
        use_fp16=use_fp16,
        device=torch.device(f'cuda:{gpu_id}')
    )
    print(f"measuring conv, fp16 enabled={use_fp16}")
    # conv_measure.run(100_000, dx=True, use_fp16=use_fp16, filename=f'conv_data_{"fp16_"}{gpu_id}.data')
    filename = f"conv_data_{device_type}_{'fp16_' if use_fp16 else ''}{gpu_id}.data"
    filename = os.path.join(args.data_dir, filename)
    conv_measure.run(100_000, dx=True, use_fp16=use_fp16, filename=filename)

def mp_measure_matmul(gpu_id, device_type):
    matmul_measure = MatMulMeasure(
        n_range=(1, 1024),
        m_range=(1, 32768),
        k_range=(1, 32768),
        device=torch.device(f'cuda:{gpu_id}')
    )
    print("measuring matmul")
    matmul_measure.run(10_000, filename=f'matmul_data_{gpu_id}.data')

def mp_measure_batchnorm(gpu_id, device_type):
    batchnorm_measure = BatchNormMeasure(
        batch_size_range=(1, 64),
        image_size_range=(2, 224),
        channels_range=(1, 1024),
        device=torch.device(f'cuda:{gpu_id}')
    )
    print("measuring batchnorm")
    batchnorm_measure.run(10_000, filename=f'batchnorm_data_{gpu_id}.data')

def mp_measure_maxpool(gpu_id, device_type):
    pool_measure = MaxPoolingMeasure(
        batch_size_range=(1, 64),
        image_size_range=(2, 256),
        channels_range=(1, 1024),
        kernel_size_range=(1, 7),
        stride_range=(1, 7),
        device=torch.device(f'cuda:{gpu_id}')
    ) 
    print("measuring maxpool")
    pool_measure.run(10_000, filename=f'maxpool_data_{gpu_id}.data')

def mp_measure(func, device_type, num_gpus=4, *args, **kwargs):
    if num_gpus == 1:
        func(0, device_type, *args, **kwargs)
    else:
        processes = [Process(target=func, args=(gpu_id, device_type) + args, kwargs=kwargs) for gpu_id in range(num_gpus)]
        for p in processes:
            p.start()
        while True:
            for i in range(len(processes)):
                processes[i].join(60)
                if processes[i].is_alive():
                   processes[i].terminate() 
                   time.sleep(1)
                processes[i] = Process(target=func, args=(i, device_type) + args, kwargs=kwargs)
                processes[i].start()
            

        for p in processes:
            p.join()

if __name__ == '__main__':
    # mp_measure(mp_measure_batchnorm, num_gpus=4)
    if args.op == "conv2d":
        mp_measure(mp_measure_conv, args.device, num_gpus=args.num_gpus, use_fp16=args.use_fp16)
    else:
        raise RuntimeError("Not supported")

