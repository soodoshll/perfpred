import torch
from torch import nn 
import time
# from vgg import build_vgg_model
# from apex import amp 
import pickle
# import torchvision
# from predictor import load_model, Conv2DPredictor
from torch.profiler import profile, record_function, ProfilerActivity
# torch.backends.cudnn.benchmark = True
# linear_pred, conv_pred, maxpool_pred = load_model()
# conv_pred = Conv2DPredictor(True)
# conv_pred.load_model("./model/predictor_model_conv2d.th")

from perfpred.utils import timing

device = torch.device('cuda')

# def load_data(filenames):
#     rows = []
#     for fn in filenames:
#         with open(fn, "rb") as f:
#             while 1:
#                 try:
#                     objs = pickle.load(f)
#                     for obj in objs:
#                         dur_forward, dur_backward, dx, use_fp16, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = obj
#                         rows.append(
#                             (0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, use_fp16, dur_forward)
#                         )
#                         rows.append(
#                             (0, batch_size, 
#                             image_size, in_channels, out_channels, kernel_size, stride, padding, 0, use_fp16, dur_backward)
#                         )
#                 except (EOFError):
#                     break
#     return rows

# data = load_data(["data/eco-18/conv_data_2080ti_fp16_0.data"])
# data = data[-30:]
# print(data)
# batch_size = 32
# in_channels = 64
# image_size = 224
warm_up = 100
nitr = 100

# out_channels = 64
# model = nn.Conv2d(in_channels, out_channels, 3, bias=False)
# # model = torchvision.models.resnet18()
# model.to(device)
# x = torch.rand((batch_size, in_channels, image_size, image_size), device=device)
# model(x)

# # model, optim = amp.initialize(model, optim, opt_level="O3")
# scaler = torch.cuda.amp.GradScaler()

def analyze(prof):
    events = prof.profiler.function_events
    for evt in events:
        if evt.name == "aten::linear":
            print('forward', evt.cuda_time_total/1e3)
        elif evt.name == "aten::linear":
            print('backward', evt.cuda_time_total/1e3)

scaler = torch.cuda.amp.GradScaler()
# for d in data:

# print(d)
# pred = conv_pred.predict(
#     [0, 2048, 3072, 768, 1, 1]
# )    
# pred += conv_pred.predict(
    # [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 0, 1]
# )    
x = torch.rand((2048, 3072), device=device, dtype=torch.float16)
model = nn.Linear(3072, 768, bias=False, device=device)
# model(x)
with profile(activities=[
    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=analyze) as prof:

    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    def foo():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(x)
            loss = out.sum()
        scaler.scale(loss).backward()
        # scaler.step(optim)
        # scaler.update()
        # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
            # scaled_loss.backward()
        # optim.step()
        torch.cuda.synchronize()
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
    # dur = timing(lambda: model(x), nitr)
        dur_tc = timing(foo, warm_up, nitr)
# print(batch_size, dur, dur_tc)
print(dur_tc)

prof.export_chrome_trace("trace_tc.json")
