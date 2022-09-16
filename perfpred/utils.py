import torch
import torchvision
from tqdm import trange

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


def change_inplace_to_false(module):
    if hasattr(module, 'inplace'):
        module.inplace = False

def warmup(device):
    default = [32, 224, 64, 64, 3, 1, 1]
    batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding = default
    print("warm up")
    warmup = 10_000
    x = torch.rand(batch_size, in_channels, image_size, image_size, device=device)
    model = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, device=device)
    for _ in trange(warmup):
        model(x)
    torch.cuda.synchronize(device)