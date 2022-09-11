import torch
from torch import nn 
import time
from vgg import build_vgg_model
# from apex import amp 
import pickle
import torchvision
from predictor import load_model, Conv2DPredictor
# linear_pred, conv_pred, maxpool_pred = load_model()
conv_pred = Conv2DPredictor(True)
conv_pred.load_model("./predictor_model_conv2d.th")

from utils import timing

device = torch.device('cuda')

def load_data(filenames):
    rows = []
    for fn in filenames:
        with open(fn, "rb") as f:
            while 1:
                try:
                    objs = pickle.load(f)
                    for obj in objs:
                        dur_forward, dur_backward, dx, use_fp16, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = obj
                        rows.append(
                            (0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, use_fp16, dur_forward)
                        )
                        rows.append(
                            (0, batch_size, 
                            image_size, in_channels, out_channels, kernel_size, stride, padding, 0, use_fp16, dur_backward)
                        )
                except (EOFError):
                    break
    return rows

data = load_data(["data/eco-13/conv_data_2080ti_fp16_1.data"])
data = data[-30:]
print(data)
# batch_size = 32
# in_channels = 64
# image_size = 224
warm_up = 10
nitr = 20

# out_channels = 64
# model = nn.Conv2d(in_channels, out_channels, 3, bias=False)
# # model = torchvision.models.resnet18()
# model.to(device)
# x = torch.rand((batch_size, in_channels, image_size, image_size), device=device)
# model(x)
# optim = torch.optim.SGD(model.parameters(), lr=1e-3)

# # model, optim = amp.initialize(model, optim, opt_level="O3")
# scaler = torch.cuda.amp.GradScaler()

for d in data:
    _, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, is_forward, use_fp16, dur = d
    # print(d)
    if not is_forward:
        continue
    pred = conv_pred.predict(
        [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 1]
    )    
    pred_32 = conv_pred.predict(
        [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 0]
    )   
    x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, dtype=torch.float16)
    model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    def foo():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(x)
            # loss = out.sum()
        # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
            # scaled_loss.backward()
        # optim.step()
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
    # dur = timing(lambda: model(x), nitr)
        dur_tc = timing(foo, nitr)
    # print(batch_size, dur, dur_tc)
    print(batch_size, dur, dur_tc, pred, pred_32)