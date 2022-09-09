import torch
from predictor import load_model
from torch import nn
torch.backends.cudnn.benchmark = True
linear_pred, conv_pred, maxpool_pred = load_model()
device = torch.device('cuda')

scaler = torch.cuda.amp.GradScaler()
from utils import timing

image_size = 224
in_channels = 64
out_channels = 64
kernel_size = 3
stride = 1
padding = 1
nitr = 20
warm_up = 5
 
for batch_size in range(1, 48):
    x = torch.rand((batch_size, in_channels, image_size, image_size), device=device, requires_grad=True, dtype=torch.float16)
    model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, device=device)
    grad = torch.ones_like(x)
    def foo():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(x)
            out.backward(grad)
            # out = out.sum()
            # scaler.scale(out).backward()
            # loss = out.sum()
        # with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
        # scaled_loss.backward()
        # optim.step()
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        for _ in range(warm_up):
            foo()
        torch.cuda.synchronize()

    # dur = timing(lambda: model(x), nitr)
    dur_tc = timing(foo, nitr)
    # print(batch_size, dur, dur_tc)
    pred = conv_pred.predict(
        [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, 1]
    ) 
    pred += conv_pred.predict(
        [0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 0, 1]
    ) 
    print(f"{batch_size}, {dur_tc}, {pred}")

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