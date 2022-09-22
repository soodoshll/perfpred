import torch
from perfpred.predictor import Conv2DPredictor
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from perfpred.utils import timing, get_clock, timing_cpu
import torchvision
torch.backends.cudnn.benchmark = True
# linear_pred, conv_pred, maxpool_pred = load_model()
# conv_pred = Conv2DPredictor()
# conv_pred.load_model("./model/predictor_model_conv2d.th")

device = torch.device('cuda')

scaler = torch.cuda.amp.GradScaler()

image_size = 224
in_channels = 3
out_channels = 64
kernel_size = 3
stride = 1
padding = 1
nitr = 10
warm_up = 10
batch_size = 32


for batch_size in range(8, 9):
    x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, dtype=torch.float32)
    model = torchvision.models.resnet50()
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    labels = torch.randint(1000 - 1, (batch_size, ), device=device)
    # model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    # grad = torch.ones_like(x)
    def foo():
        # out = out.sum()
        # return out
        # out.backward()
        torch.cuda.synchronize()
        optim.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(x)
            loss = loss_fn(out, labels)
            # out.backward()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()
            # loss = out.sum()
        # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
        # scaled_loss.backward()
        # optim.step()
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]) as p:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            dur_tc = timing_cpu(foo, warm_up, nitr)

        torch.cuda.synchronize()
    p.export_chrome_trace("trace_tc_8.json")
        # clock = get_clock()

    # prof.export_chrome_trace("trace_tc.json")
        # dur_tc = timing(foo, nitr)
    # print(batch_size, dur, dur_tc)
    # pred = conv_pred.predict(
    #     [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 0]
    # ) 
    # pred_fp32 = conv_pred.predict(
    #     [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 0]
    # ) 
    # pred += conv_pred.predict(
        # [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 0, 1]
    # ) 
    print(f"{batch_size}, {dur_tc}")

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