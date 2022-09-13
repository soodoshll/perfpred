import torch
from predictor import Conv2DPredictor
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

torch.backends.cudnn.benchmark = True
# linear_pred, conv_pred, maxpool_pred = load_model()
conv_pred = Conv2DPredictor()
conv_pred.load_model("./model/predictor_model_conv2d.th")

device = torch.device('cuda')

# scaler = torch.cuda.amp.GradScaler()
from utils import timing

image_size = 224
in_channels = 64
# out_channels = 64
kernel_size = 3
stride = 1
padding = 1
nitr = 10
warm_up = 3
batch_size = 32

for out_channels in range(4, 515, 4):
    x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, dtype=torch.float16)
    model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    # grad = torch.ones_like(x)
    def foo():
        out = model(x)
        # out = out.sum()
        # return out
        # out.backward()
        torch.cuda.synchronize()
            # out = out.sum()
            # scaler.scale(out).backward()
            # loss = out.sum()
        # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
        # scaled_loss.backward()
        # optim.step()
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        dur_tc = timing(foo, warm_up, nitr)

    # prof.export_chrome_trace("trace_tc.json")
        # dur_tc = timing(foo, nitr)
    # print(batch_size, dur, dur_tc)
    pred = conv_pred.predict(
        [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 1]
    ) 
    # pred_fp32 = conv_pred.predict(
    #     [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 0]
    # ) 
    # pred += conv_pred.predict(
        # [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 0, 1]
    # ) 
    print(f"{out_channels}, {dur_tc}, {pred}")

exit()

x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, requires_grad=True, dtype=torch.float16)
model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
# x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, requires_grad=True, dtype=torch.float16)

def foo():
    out = model(x)
        # loss = out.sum()
    # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
        # scaled_loss.backward()
    # optim.step()
# with torch.autocast(device_type='cuda', dtype=torch.float16):
with torch.autocast(device_type='cuda', dtype=torch.float16):
    for _ in range(5):
        foo()
    torch.cuda.synchronize()
    dur = timing(foo, 20)

print(dur)



# print(f"{bs}, {dur}, {pred}")