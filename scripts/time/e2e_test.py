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
torch_vision_model_revise()

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

        # if args.verbose >= 1:
        #     for t_item, pred_module_dur in zip(trace_with_dur, pred_dur):
        #         is_forward, module, _, _, dur = t_item
        #         if isinstance(module, nn.Conv2d):
        #             print(module)
        #             print(f'{is_forward}, {str(type(module))[25:-2]}, {pred_module_dur}, {dur/1e3}')
        
        # data_bs.append([pred, dur_measure, truth_kernel_time])
        print(pred)
        del inputs, labels 

# if args.plot:
#     data = np.array(data)

#     # paint
#     w = 0.3

#     def autolabel(ax, data):
#         bar_num = len(ax.patches)
#         for bar_id in range(bar_num // 2):
#             rect1 = ax.patches[bar_id ]
#             rect2 = ax.patches[bar_id + bar_num // 2]
#             x = (rect1.get_x() + rect2.get_x()) / 2 + w / 2
#             y = max(rect1.get_height(), rect2.get_height())
#             ax.annotate(f"{data[bar_id] * 100: .2f}%", (x,y), xytext=(0,5), textcoords="offset points",
#                         ha='center', va='bottom')

#     plt.figure()
#     fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 6),)
#     for i in range(3):
#         ax = axes[i]
#         x = np.arange(1, 3)
#         labels = ['fp32', 'fp16']
#         err = abs(data[i, :, 0] - data[i, :, 1]) / data[i, :, 1]
#         ax.bar(x-w/2, data[i, :, 0], width=w, edgecolor='black', label='pred')
#         ax.bar(x+w/2, data[i, :, 1], width=w, edgecolor='black', label='truth')
#         ax.set_xlabel("batch size =" + str(args.batch_size[i]))
#         autolabel(ax, err)
#         ax.set_xticks(x, labels)

#     axes[0].set_ylabel('time (ms)')
#     ax.legend()
#     fig.subplots_adjust(wspace=0)
#     plt.savefig(f"figure/e2e_{args.model}_{args.nomodulo}_error.png")