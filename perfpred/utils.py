import torch
import torchvision
import time
import subprocess
import os
from torch.autograd import DeviceType

def timing(func, warmup=3, nitr=20, verbose=0):
    if verbose >= 1:
        print("warmup...")
        for _ in range(warmup):
            func()
    else:
        for _ in range(warmup):
            func()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nitr):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / nitr

def timing_cpu(func, warmup=3, nitr=20, verbose=0):
    if verbose >= 1:
        print("warmup...")
        for _ in range(warmup):
            func()
    else:
        for _ in range(warmup):
            func()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(nitr):
        func()
    torch.cuda.synchronize()
    return (time.time() - start) / nitr * 1e3 

def torch_vision_model_revise():
    def _basicblock_revised_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

    def _bottleneck_revised_forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

    torchvision.models.resnet.BasicBlock.forward = _basicblock_revised_forward
    torchvision.models.resnet.Bottleneck.forward = _bottleneck_revised_forward

    def _basic_conv2d_revised_forawrd(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.nn.functional.relu(x)
    torchvision.models.inception.BasicConv2d.forward = _basic_conv2d_revised_forawrd

def change_inplace_to_false(module):
    if hasattr(module, 'inplace'):
        module.inplace = False

def warmup(device):
    default = [32, 224, 64, 64, 3, 1, 1]
    batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding = default
    print("warm up")
    warmup = 1_000
    x = torch.rand(batch_size, in_channels, image_size, image_size, device=device)
    model = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=device)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize(device)

def get_clock():
    out = subprocess.run("nvidia-smi -q -d CLOCK -i 0", capture_output=True, shell=True)
    for line in out.stdout.split(b'\n'):
        if line.strip().startswith(b'SM'):
            clock = (int(line.split()[-2]))
            break
    return clock

def remove_initialization():
    initializers = [
        "xavier_normal", 
        "xavier_uniform",
        "normal",
        "trunc_normal",
        "uniform",
        "zeros",
        "eye",
        "constant",
        "ones",
        "dirac",
        "kaiming_uniform",
        "kaiming_normal",
        "orthogonal"
    ]

    for init in initializers:
        exec(f"torch.nn.init.{init} = lambda *args, **kwargs: None")
        exec(f"torch.nn.init.{init}_ = lambda *args, **kwargs: None")
    
    exec("torch.Tensor.normal_ = lambda *args, **kwargs: None")

def parse_mem_log(filename, pid):
    max_mem = 0
    with open(filename, "r") as f:
        for line in f:
            tokens = line.split(',')
            if tokens[0] == str(pid):
                mem = int(tokens[1].split()[0])
                max_mem = max(max_mem, mem)
    return max_mem

def measure_gpu_mem(func):
    if os.path.isfile("log_mem.tmp"):
        os.remove("log_mem.tmp")
    monitor_proc = subprocess.Popen(['nvidia-smi', '-lms', '5', '--query-compute-apps=pid,used_gpu_memory', '--format=csv', '-f', 'log_mem.tmp'])
    func()
    monitor_proc.terminate()
    time.sleep(1)
    max_mem = parse_mem_log("log_mem.tmp", os.getpid())
    os.remove("log_mem.tmp")
    return max_mem

def _get_all_children(events, root):
    ret = []
    start_idx = events.index(root)
    for evt in events[start_idx:]:
        if evt.time_range.start >= root.time_range.start and evt.time_range.end <= root.time_range.end:
            ret.append(evt)
        if evt.time_range.start > root.time_range.end:
            break
    return ret

def _get_first_level_ops(trace, root):
    children = _get_all_children(trace, root)
    first_level_ops = []
    for evt in children:
        if evt.device_type == DeviceType.CPU and (evt.cpu_parent is None or evt.cpu_parent == root): # first level operators
            first_level_ops.append(evt)
    return first_level_ops

def profile_model(func, nitr=20, device='cuda', dump_file=None):
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        schedule= torch.profiler.schedule(
            wait=1,
            warmup=5,
            active=nitr,
            repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # with_stack=True,
        # with_modules=True,
        record_shapes=True,
    ) as profiler:
        for _ in range(nitr + 7):
            func()
            torch.cuda.synchronize(device)
            profiler.step()
        torch.cuda.synchronize(device)
    if dump_file is not None:
        profiler.export_chrome_trace(dump_file)
    return profiler.profiler.function_events