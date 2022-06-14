import torch
import time
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from dataloading import *
import numpy as np
from torch.autograd import DeviceType

torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0')
torch.cuda.set_device(device)

conv_pred = Conv2DPredictor()
# CONV2D_PATH = glob.glob("./data_backup/conv_data_*.data")
# conv_pred.load_data(CONV2D_PATH)
conv_pred.load_data_sql(CONV2D_PATH_SQL)
conv_pred.load_model("predictor_model_conv2d.th")

nitr = 1000
dataset = conv_pred.raw_dataset
record = []

def analyze(prof):
    events = prof.profiler.function_events
    for evt in events:
        if evt.device_type == DeviceType.CPU:
            duration = sum([k.duration for k in evt.kernels]) / 1e3
            if evt.name == "aten::cudnn_convolution":
                # print(duration)
                record.append(duration)

for i in range(nitr):
    record = []
    while True:
        idx = 2
        # idx = random.randint(0, dataset.shape[0] - 1)
        bias, batch, image_size, in_channels, out_channels, kernel_size, stride, padding, is_forward = dataset[idx, :-1].numpy().astype(int)
        run_time = dataset[idx, -1].numpy()
        if is_forward:
            break
    # idx=0
    # print(bias, batch, image_size, in_channels, out_channels, kernel_size, padding, stride, is_forward, run_time)
    input = torch.rand((batch, in_channels, image_size, image_size), device=device)
    layer = nn.Conv2d(in_channels, out_channels, padding=padding, stride=stride, kernel_size=kernel_size, device=device, bias=False)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        on_trace_ready=analyze
    ) as profiler:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            out = layer(input)
        torch.cuda.synchronize()
        dur_py = (time.time()-t0)/5 * 1e3
    
    dur = np.mean(record)
    print(dur, dur_py, run_time)

