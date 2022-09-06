import torch
import time
import utils
import torch.nn.functional as F
import functools

# Setup initial tensors and parameters
input_size = [32, 3, 224, 224]
device = "cuda"
dtype = torch.float32

# Create sample inputs
input1 = torch.randn(*input_size, device=device, dtype=dtype)
conv = torch.nn.Conv2d(3, 64, 3, device=device)
bn = torch.nn.BatchNorm2d(64, device=device)
bn.eval()
bias = torch.nn.Parameter(torch.zeros(64,1,1, device=device))

def conv_bn_relu(input, weight, bias, running_mean, running_var):
    out = F.conv2d(input, weight)
    out = out + bias
    out = F.batch_norm(out, running_mean, running_var)
    # out = bn(out)
    # out = input + 1
    # out = input
    out = F.relu(out)
    return out

# Utility to profile the workload
def profile_workload(forward_func, iteration_count=100, label=""):
    # Perform warm-up iterations
    for _ in range(3):
        output = forward_func()

    # Synchronize the GPU before starting the timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    for _ in range(iteration_count):
        output = forward_func()
    end.record()

    torch.cuda.synchronize()
    # stop = time.perf_counter()
    print("Duration:", start.elapsed_time(end) / iteration_count)

func = functools.partial(
    conv_bn_relu,
    input1,
    conv.weight,
    bias,
    bn.running_mean,
    bn.running_var
)
profile_workload(
    func, iteration_count=100, label="Eager Mode - Composite definition"
)

scripted_composite_definition = torch.jit.script(conv_bn_relu)
func = functools.partial(
    scripted_composite_definition,
    input1,
    conv.weight,
    bias,
    bn.running_mean,
    bn.running_var
)

profile_workload(
    func, iteration_count=100, label="TorchScript - Composite definition"
)

print(scripted_composite_definition.graph_for(input1))