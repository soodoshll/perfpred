import torch
import torchvision
from tqdm import trange
import time
import subprocess

def timing(func, warmup=3, nitr=20, verbose=0):
    if verbose >= 1:
        print("warmup...")
        for _ in trange(warmup):
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
        for _ in trange(warmup):
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
    for _ in trange(warmup):
        model(x)
    torch.cuda.synchronize(device)

def get_clock():
    out = subprocess.run("nvidia-smi -q -d CLOCK -i 0", capture_output=True, shell=True)
    for line in out.stdout.split(b'\n'):
        if line.strip().startswith(b'SM'):
            clock = (int(line.split()[-2]))
            break
    return clock