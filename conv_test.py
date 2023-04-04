import torch
from perfpred.predictor import Conv2DPredictor
from torch import nn
from perfpred.utils import timing, get_clock, timing_cpu
torch.backends.cudnn.benchmark = False

device = torch.device('cuda')

image_size = 112
in_channels = 64
out_channels = 64
kernel_size = 3
stride = 1
padding = 1
nitr = 200
warm_up = 200
batch_size = 32

conv_pred = Conv2DPredictor()
conv_pred.load_model("./model/2080ti/predictor_model_conv2d.th")

for batch_size in [32]:
    x = torch.empty((batch_size, in_channels, image_size, image_size), device=device, dtype=torch.float32)
    model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    pred = conv_pred.predict([0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 0])
    def foo():
        torch.cuda.synchronize()
        out = model(x)
        torch.cuda.synchronize()

    dur_tc = timing_cpu(foo, warm_up, nitr)
    print(f"{batch_size}, {dur_tc}, {pred}")