import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import DeviceType
import functools
from functools import partial
import numpy as np
# from tqdm import tqdm, trange
import pickle

import argparse
import random
import time
import os

import multiprocessing as mp
from multiprocessing import Process

# torch.set_grad_enabled(True)
torch.backends.cudnn.benchmark = True

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

def measure_op(inputs_generator, measured_func, analyze_func, device, use_amp=False, nitr=3, cooldown=0):
    data = inputs_generator()
    info = {}
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        schedule= torch.profiler.schedule(
            wait=1,
            warmup=3,
            active=nitr,
            repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        on_trace_ready=functools.partial(analyze_func, info)
    ) as profiler:
        for _ in range(nitr + 5):
            measured_func(data)
            profiler.step()
            time.sleep(cooldown)
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
        for child in event.cpu_children:
            get_children_kernel_start_and_end(child, start_list, end_list)
    start_list, end_list = [], []
    get_children_kernel_start_and_end(event, start_list, end_list)
    if len(start_list) == 0 or len(end_list) == 0:
        return 0
    start = min(start_list)
    end = max(end_list)
    return end - start

class Measure(object):
    def get_measured_func(self):
        def fun(data):
            x = data[0]
            model = data[1]
            o = model(x).sum()
            o.backward()
            torch.cuda.synchronize(self.device)
        
        def fun_fp16(data):
            x = data[0]
            model = data[1]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                o = model(x).sum()
            self.scaler.scale(o).backward()
            torch.cuda.synchronize(self.device)            
        return fun_fp16 if self.use_amp else fun

    def run(self, step=1, dx=True, use_amp=False, filename='data', cooldown=0):
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        with open(filename, 'ab+') as f:
            for _ in trange(step):
                success = False
                while not success:
                    success = True
                    try:
                        ret = measure_op(self.get_inputs_generator(), self.get_measured_func(), self.get_analyze_func(), 
                        device=self.device, use_amp = use_amp, cooldown=cooldown) 
                    except RuntimeError as e:
                        success = False
                pickle.dump(ret['data'], f)
                f.flush()

class MatMulMeasure(Measure):
    forward_op_kw = ["aten::linear"]
    backward_op_kw = ["AddmmBackward0", "MmBackward0"]
    def __init__(self, n_range, m_range, k_range, device):
        self.n_range = n_range
        self.m_range = m_range
        self.k_range = k_range
        self.data = []
        self.record = []
        self.device = device
    
    def get_inputs_generator(self):
        bias = random.choice([True, False])
        n = random.randint(*self.n_range) 
        m = random.randint(*self.m_range)
        k = random.randint(*self.k_range)
        self.params = (n, m, k, bias)
        dtype = torch.float16 if self.use_amp else torch.float32
        def fun():
            x = torch.empty((n, m), device=self.device, dtype=dtype, requires_grad=True)
            model = torch.nn.Linear(m, k, device=self.device, bias=bias)
            return x, model
        return fun

    def get_analyze_func(self):
        def fun(info, prof):
            events = prof.profiler.function_events
            tmp_forward_durations = []
            tmp_backward_durations = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                if evt.device_type == DeviceType.CPU and evt.name in self.forward_op_kw:
                    duration = get_children_kernel_time(evt) / 1e3
                    tmp_forward_durations.append(duration)
                if evt.device_type == DeviceType.CPU and evt.name in self.backward_op_kw:
                    duration = get_children_kernel_time(evt) / 1e3
                    tmp_backward_durations.append(duration)
            dur_avg_forward = np.mean(tmp_forward_durations)
            dur_avg_backward = np.mean(tmp_backward_durations)
            # print(dur_avg_forward, dur_avg_backward)
            info['data'].append((dur_avg_forward, dur_avg_backward, self.use_amp) + self.params)      
        return fun
    
    def numpy(self):
        return np.array(self.record)

class BatchMatMulMeasure(Measure):
    forward_op_kw = "aten::bmm"
    backward_op_kw = "BmmBackward0"
    def __init__(self, batch_size_range, l_range, m_range, n_range, use_amp=True, device='cpu'):
        self.batch_size_range = batch_size_range
        self.l_range = l_range
        self.m_range = m_range
        self.n_range = n_range
        self.use_amp = use_amp
        self.record = []

        self.device = device
    
    def get_inputs_generator(self):
        batch_size = random.randint(*self.batch_size_range)
        l = random.randint(*self.l_range)
        m = random.randint(*self.m_range)
        n = random.randint(*self.n_range)
        self.params = (batch_size, l, m, n)
        def func():
            dtype = torch.float16 if self.use_amp else torch.float32
            A = torch.rand((batch_size, l, m), device=self.device, requires_grad=True, dtype=dtype)
            B = torch.rand((batch_size, m, n), device=self.device, requires_grad=True, dtype=dtype)
            return A, B
        return func

    def get_analyze_func(self):

        def fun(info, prof):
            events = prof.profiler.function_events
            tmp_forward_durations = []
            tmp_backward_durations = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                if evt.device_type == DeviceType.CPU and evt.name in self.forward_op_kw:
                    # duration = evt.cpu_parent.cuda_time_total / 1e3
                    duration = evt.cuda_time_total / 1e3
                    tmp_forward_durations.append(duration)
                if evt.device_type == DeviceType.CPU and evt.name in self.backward_op_kw:
                    duration = evt.cuda_time_total / 1e3
                    tmp_backward_durations.append(duration)
            dur_avg_forward = np.mean(tmp_forward_durations)
            dur_avg_backward = np.mean(tmp_backward_durations)
            info['data'].append((dur_avg_forward, dur_avg_backward, self.use_amp) + self.params)      
        return fun
    
    def get_measured_func(self):
        def func(data):
            A = data[0]
            B = data[1]
            out = A.bmm(B)
            out = out.sum()
            out.backward()
            torch.cuda.synchronize(self.device)

        def func_fp16(data):
            A = data[0]
            B = data[1]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = A.bmm(B)
                out = out.sum()
            self.scaler.scale(out).backward()
            torch.cuda.synchronize(self.device)
        return func_fp16 if self.use_amp else func

class ConvMeasure(Measure):
    """
    Measure forward and backward for conv2D
    """
    forward_name = "aten::cudnn_convolution"
    backward_names = ["aten::convolution_backward"]

    def __init__(self, batch_size_range, image_size_range, 
                 in_channels_range, out_channels_range, kernel_size_range, stride_range,
                 padding_range, use_amp=True, device=torch.device('cpu')):
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
        def func():
            dtype = torch.float16 if self.use_amp else torch.float32
            A = torch.rand((batch_size, in_channels, image_size, image_size), device=self.device, requires_grad=True, dtype=dtype)
            layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, device=self.device, bias=False)
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
        return func_fp16 if self.use_amp else func
    
    def get_analyze_func(self):
        def func(info, prof):
            events = prof.profiler.function_events
            tmp_dur_forward = []
            tmp_dur_backward = []
            if not 'data' in info.keys():
                info['data'] = []
            for evt in events:
                if evt.device_type == DeviceType.CPU:
                    duration = evt.cuda_time_total / 1e3
                    if evt.name in self.backward_names:
                        # print(evt.name)
                        tmp_dur_backward.append(duration)
                        backward_shape = evt.input_shapes
                    elif evt.name == self.forward_name:
                        tmp_dur_forward.append(duration)
                        forward_shape = evt.input_shapes
            
            dur_avg_forward = np.mean(tmp_dur_forward)
            dur_avg_backward = np.mean(tmp_dur_backward)
            info['data'].append((dur_avg_forward, dur_avg_backward, self.dx, self.use_amp) + self.params)      
        return func


    def numpy(self):
        return np.array(self.record_forward) #, np.array(self.record_dw), np.array(self.record_dwdx)

class BatchNormMeasure(Measure):
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

        def func_fp16(data):
            A = data[0]
            layer = data[1]
            layer.train()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = layer(A)
                out = out.sum()
            self.scaler.scale(out).backward()
            torch.cuda.synchronize(self.device)

        return func_fp16 if self.use_amp else func

    def get_inputs_generator(self): 
        batch_size = random.randint(*self.batch_size_range)
        image_size = random.randint(*self.image_size_range)
        channels = random.randint(*self.channels_range)
        self.params = (batch_size, image_size, channels)
        dtype = torch.float16 if self.use_amp else torch.float32
        def func(dx=True):
            self.dx = dx
            A = torch.rand((batch_size, channels, image_size, image_size), device=self.device, requires_grad=dx, dtype=dtype)
            layer = torch.nn.BatchNorm2d(channels, device=self.device)
            layer.train()
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
            info['data'].append((dur_avg_forward, dur_avg_backward, self.use_amp) + self.params)      
        return func

    def run(self, step=1, dx=True, filename='batchnorm_data', use_amp=False):
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        with open(filename, 'ab+') as f:
            for _ in trange(step):
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

class MaxPoolingMeasure(Measure):
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
        dtype = torch.float16 if self.use_amp else torch.float32
        def func():
            A = torch.rand((batch_size, channels, image_size, image_size), device=self.device, requires_grad=True, dtype=dtype)
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
        
        def func_fp16(data):
            A = data[0]
            layer = data[1]
            layer.train()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = layer(A)
                out = out.sum()
            self.scaler.scale(out).backward()
            torch.cuda.synchronize(self.device)

        return func_fp16 if self.use_amp else func 

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
            info['data'].append((dur_avg_forward, dur_avg_backward, self.use_amp) + self.params)      
        return func 

    def run(self, step=1, filename='maxpool_data', use_amp=False):
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler() 
        with open(filename, 'ab+') as f:
            for _ in range(step):
                success = False
                while not success:
                    success = True
                    try:
                        ret = measure_op(partial(self.get_inputs_generator()), self.get_measured_func(), self.get_analyze_func(), device=self.device) 
                    except RuntimeError as e:
                        if str(e).find('out of memory') >= 0:
                            success = False
                        else:
                            return
                pickle.dump(ret['data'], f)
                f.flush()    

    def numpy(self):
        return np.array(self.record_forward), np.array(self.record_backward)

def _data_filename(data_dir, op, gpu_id, device_type, use_amp):
    filename = f"{op}_{device_type}_{'amp_' if use_amp else ''}{gpu_id}.data"
    filename = os.path.join(data_dir, filename)
    return filename

def mp_measure_conv(gpu_id, args):
    max_batch_size = 48 if args.device == "2070" else 64
    conv_measure = ConvMeasure(
        batch_size_range=(1, max_batch_size),
        image_size_range=(2, 224),
        in_channels_range=(3, 1024),
        out_channels_range=(16, 1024),
        kernel_size_range=(1, 7),
        stride_range=(1, 7),
        padding_range=(0, 3),
        use_amp=args.use_amp,
        device=torch.device(f'cuda:{gpu_id}')
    )
    print(f"measuring conv, fp16 enabled={args.use_amp}")
    filename = _data_filename(args.data_dir, "conv", gpu_id, args.device, args.use_amp)
    conv_measure.run(100_000, dx=True, use_amp=args.use_amp, filename=filename, cooldown=args.cooldown)

def mp_measure_matmul(gpu_id, args):
    matmul_measure = MatMulMeasure(
        n_range=(1, 8192),
        m_range=(1, 32768),
        k_range=(1, 32768),
        device=torch.device(f'cuda:{gpu_id}')
    )
    print("measuring matmul")
    filename = _data_filename(args.data_dir, "matmul", gpu_id, args.device, args.use_amp)
    matmul_measure.run(10_000, use_amp=args.use_amp, filename=filename)

def mp_measure_bmm(gpu_id, args):
    bmm_measure = BatchMatMulMeasure(
        batch_size_range=(1, 129),
        l_range=(1, 1025),
        m_range=(1, 1025),
        n_range=(1, 1025),
        device=torch.device(f'cuda:{gpu_id}')
    )
    print("measuring bmm")
    filename = _data_filename(args.data_dir, "bmm", gpu_id, args.device, args.use_amp)
    bmm_measure.run(10_000, use_amp=args.use_amp, filename=filename)

def mp_measure_batchnorm(gpu_id, args):
    batchnorm_measure = BatchNormMeasure(
        batch_size_range=(1, 64),
        image_size_range=(2, 224),
        channels_range=(1, 1024),
        device=torch.device(f'cuda:{gpu_id}')
    )
    print("measuring batchnorm")
    filename = _data_filename(args.data_dir, "batchnorm", gpu_id, args.device, args.use_amp)
    batchnorm_measure.run(10_000, use_amp=args.use_amp, filename=filename)

def mp_measure_maxpool(gpu_id, args):
    pool_measure = MaxPoolingMeasure(
        batch_size_range=(1, 64),
        image_size_range=(2, 256),
        channels_range=(1, 1024),
        kernel_size_range=(1, 7),
        stride_range=(1, 7),
        device=torch.device(f'cuda:{gpu_id}')
    ) 
    print("measuring maxpool")
    filename = _data_filename(args.data_dir, "maxpool", gpu_id, args.device, args.use_amp)
    pool_measure.run(10_000, use_amp=args.use_amp, filename=filename)

def mp_measure(func, args):
    """
    fp16 training is not so stable and i don't know why
    so just periodically restart them to avoid being blocked
    and i can sleep well without manually checking if the program is still running
    """
    
    if args.num_gpus == 0:
        func(0, args)
    else:
        mp.set_start_method('spawn')
        processes = [Process(target=func, args=(gpu_id, args)) for gpu_id in range(args.num_gpus)]
        for p in processes:
            p.start()
        while True:
            for i in range(len(processes)):
                processes[i].join(60)
                if processes[i].is_alive():
                    processes[i].terminate()
                    processes[i].join()
                print("restart process", i)
                processes[i] = Process(target=func, args=(i, args))
                processes[i].start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=["conv2d", "mm", "batchnorm", "maxpool2d", "bmm"])
    parser.add_argument("device", choices=['2070', '2080ti', 't4', 'v100', '3090'])
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--cooldown", type=float, default=0.01)

    args = parser.parse_args()
    global cooldown
    cooldown = args.cooldown

    if args.op == "conv2d":
        mp_measure(mp_measure_conv, args)
    elif args.op == "mm":
        mp_measure(mp_measure_matmul, args)
    elif args.op == "batchnorm":
        mp_measure(mp_measure_batchnorm, args)
    elif args.op == "maxpool2d":
        mp_measure(mp_measure_maxpool, args)
    elif args.op == "bmm":
        mp_measure(mp_measure_bmm, args)
    else:
        raise RuntimeError("Not supported")

