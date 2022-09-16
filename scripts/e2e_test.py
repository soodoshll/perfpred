import argparse
import torch
import torchvision
from torch import nn

from perfpred.vgg import build_vgg_model
from perfpred.utils import torch_vision_model_revise, change_inplace_to_false, timing
from perfpred.trace import predict

torch_vision_model_revise()

device = torch.device('cuda')
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=int, default=0)
parser.add_argument("--model", choices=['resnet50', 'vgg'], default='vgg')
parser.add_argument("--batch_size", nargs='+', type=int, default=[8, 16, 32])
parser.add_argument("--use_fp16", action="store_true")
parser.add_argument("--nomodulo", action="store_true")
args = parser.parse_args()

print(args)

if args.model == 'vgg':
    model = build_vgg_model()
elif args.model == 'resnet50':
    model = torchvision.models.resnet50()
else:
    raise RuntimeError("not supported")

model.apply(change_inplace_to_false)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)
fp16_options = [False, True] if args.use_fp16 else [False]
for batch_size in args.batch_size:
    for use_fp16 in fp16_options:
        if use_fp16:
            scaler = torch.cuda.amp.GradScaler() 
        inputs = torch.rand([batch_size, 3, 224, 224], device=device)
        labels = torch.randint(1000 - 1, (batch_size, ), device=device)

        def trace_func():
            optim.zero_grad(set_to_none=True)
            if use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(inputs)
                    loss = loss_fn(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                out = model(inputs)
                loss = loss_fn(out, labels) 
                loss.backward()
                optim.step()
            torch.cuda.synchronize()
            del out


        dur_measure = timing(trace_func, 1_000, 100, verbose=1)
        pred, _, truth_kernel_time, trace_with_dur, pred_dur = \
            predict(model, trace_func, use_fp16=use_fp16, verbose=args.verbose)
        print(f"{batch_size}, {use_fp16}, {pred}, {dur_measure}, {truth_kernel_time}")

        if args.verbose >= 1:
            for t_item, pred_module_dur in zip(trace_with_dur, pred_dur):
                is_forward, module, _, _, dur = t_item
                if isinstance(module, nn.Conv2d):
                    # print(module)
                    print(f'{is_forward}, {str(type(module))[25:-2]}, {pred_module_dur}, {dur/1e3}')
        # if isinstance(module, nn.BatchNorm2d):
    #         print(f'{is_forward}, {str(type(module))[25:-2]}, {pred_module_dur}, {dur/1e3}')      
        # if isinstance(module, nn.Linear):
            # print(f'{is_forward}, {str(type(module))[25:-2]}, {pred_module_dur}, {dur/1e3}')
        del inputs, labels 