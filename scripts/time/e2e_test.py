import argparse
import torch
import torchvision
from torch import nn
import numpy as np

from perfpred import trace
from perfpred.predictor import Conv2DPredictor
from perfpred.utils import warmup, torch_vision_model_revise, change_inplace_to_false, timing, timing_cpu
from perfpred.trace import Predictor
# from matplotlib import pyplot as plt


torch.backends.cudnn.benchmark = True
torch_vision_model_revise(torchvision)

device = torch.device('cuda')
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=int, default=0)
parser.add_argument("--model", type=str, default='resnet18')
parser.add_argument("--batch_size", nargs='+', type=int, default=[8, 16, 32])
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--nomodulo", action="store_true")
# parser.add_argument("--plot", action="store_true")
parser.add_argument("target", type=str)
args = parser.parse_args()

print(args)

# TODO: fix this
# if args.nomodulo:
#     trace.conv_pred = Conv2DPredictor(False)
#     trace.conv_pred.load_model("./model/predictor_model_conv2d_nomodulo.th")

model = getattr(torchvision.models, args.model)()
model.apply(change_inplace_to_false)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)
amp_options = [True] if args.use_amp else [False]
first = True
data = []

if args.model == 'inception_v3':
    image_size = 299
else:
    image_size = 224

predictor = Predictor(args.target, use_amp=args.use_amp)

for batch_size in args.batch_size:
    data_bs = []
    data.append(data_bs)
    for use_amp in amp_options:
        if use_amp:
            scaler = torch.cuda.amp.GradScaler() 
        inputs = torch.rand([batch_size, 3, image_size, image_size], device=device)
        labels = torch.randint(1000 - 1, (batch_size, ), device=device)

        def trace_func():
            optim.zero_grad(set_to_none=False)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(inputs)
                    if args.model == 'inception_v3':
                        out = out[0]
                    loss = loss_fn(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                out = model(inputs)
                if args.model == 'inception_v3':
                    out = out[0]
                loss = loss_fn(out, labels) 
                loss.backward()
                optim.step()
            torch.cuda.synchronize()
            del out
        
        dur_measure = timing_cpu(trace_func, 100, 100, verbose=0)
        pred, _ = predictor.predict(model, trace_func, verbose=args.verbose)

        print(pred)
        del inputs, labels 