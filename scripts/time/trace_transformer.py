# new version of the tracer and predictor
# take transformer as an example

import torch
from torch.autograd import DeviceType

import transformers
from transformers import BertConfig, BertModel, GPT2Model, GPT2Config, BertForPreTraining, BertForSequenceClassification
import os
from ctypes import cdll
import argparse

import numpy as np

import perfpred.trace
print(perfpred.trace.UNARY_COEFF)

torch.backends.cudnn.benchmark = True
from perfpred.predictor import Conv2DPredictor, LinearPredictor, MaxPoolingPredictor, BatchNormPredictor, BatchMatMulPredict

conv_pred = Conv2DPredictor(True)
conv_pred.load_model("./model/predictor_model_conv2d.th")

linear_pred = LinearPredictor()
linear_pred.load_model("./model/predictor_model_linear.th")

maxpool_pred = MaxPoolingPredictor()
maxpool_pred.load_model("./model/predictor_model_maxpool.th")

batchnorm_pred = BatchNormPredictor()
batchnorm_pred.load_model("./model/predictor_model_batchnorm.th")

bmm_pred = BatchMatMulPredict()
bmm_pred.load_model("./model/predictor_model_bmm.th")

parser = argparse.ArgumentParser()
parser.add_argument('--amp', action='store_true')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seq_len', type=int, default=512)
args = parser.parse_args()

device = 'cuda'

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)
model.train()

optim = torch.optim.SGD(model.parameters(), lr=1e-3)
x = torch.zeros((args.batch_size, args.seq_len), device=device, dtype=torch.int32)
use_fp16 = args.amp

def trace_handler(prof):
    events = prof.profiler.function_events
    for evt in events:
        if evt.name == 'aten::bmm':
            print(evt.cpu_parent.input_shapes)

def profile_model(func, nitr=20, device='cuda'):
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        schedule= torch.profiler.schedule(
            wait=1,
            warmup=5,
            active=nitr,
            repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        # with_stack=True,
        # with_modules=True,
        record_shapes=True,
    ) as profiler:
        for _ in range(nitr + 7):
            func()
            profiler.step()
        torch.cuda.synchronize(device)
    return profiler.profiler.function_events

def _get_all_children(events, root):
    ret = []
    for evt in events:
        if evt.time_range.start >= root.time_range.start and evt.time_range.end <= root.time_range.end:
            ret.append(evt)
    return ret

def traced_func():
    torch.cuda.synchronize()
    optim.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
        out = model(x)
        loss = out['logits'].sum()
    loss.backward()
    optim.step()
    torch.cuda.synchronize()

trace = profile_model(traced_func, nitr=1)

for event in trace:
    if event.name.startswith("ProfilerStep"):
        root = event

children = _get_all_children(trace, root)
first_level_ops = []
for evt in children:
    if evt.device_type == DeviceType.CPU and (evt.cpu_parent is None or evt.cpu_parent == root): # first level operators
        first_level_ops.append(evt)

pred_tot = 0
unknown_tot = 0

linear_tot = 0
linear_pred_tot = 0

bmm_tot = 0
bmm_pred_tot = 0

skip_ops = [
    'aten::transpose', 'aten::view', 'aten::permute', 'aten::unsqueeze'
    'autograd::engine::evaluate_function: TransposeBackward0',
    'autograd::engine::evaluate_function: ViewBackward0',
    'autograd::engine::evaluate_function: PermuteBackward0',
    'autograd::engine::evaluate_function: CloneBackward0',
    'autograd::engine::evaluate_function: ExpandBackward0',
    'autograd::engine::evaluate_function: UnsafeViewBackward0',
    'autograd::engine::evaluate_function: TBackward0',
    'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad',
    'autograd::engine::evaluate_function: AddBackward0',
]

misc_tot = 0
misc_pred_tot = 0

for evt in first_level_ops:
    if evt.name in skip_ops or evt.cuda_time_total == 0 or evt == root:
        continue
    elif evt.name == 'aten::linear':
        inputs = evt.input_shapes
        bias = len(inputs) == 3
        n =  np.prod(inputs[0][:-1])
        m = inputs[0][-1]
        k = inputs[1][0]
        pred = linear_pred.predict([bias, n, m, k, 1, use_fp16])
        linear_tot += evt.cuda_time_total / 1e3
        linear_pred_tot += pred
    elif evt.name == 'autograd::engine::evaluate_function: AddmmBackward0':
        children = evt.cpu_children
        mm1 = children[0].cpu_children[1]
        assert(mm1.name == 'aten::mm')
        inputs = mm1.input_shapes
        bias = True
        n, m, k = inputs[0][0], inputs[0][1], inputs[1][1]
        pred = linear_pred.predict([bias, n, m, k, 0, use_fp16])
        linear_tot += evt.cuda_time_total / 1e3
        linear_pred_tot += pred
    elif evt.name == 'aten::matmul':
        # bmm
        for c in evt.cpu_children:
            if c.name == 'aten::bmm':
                bmm_evt = c
        inputs = bmm_evt.input_shapes
        batch_size, l, m, n = inputs[0][0], inputs[0][1], inputs[0][2], inputs[1][2]
        pred = bmm_pred.predict([batch_size, l, m, n, 1, use_fp16])
        bmm_tot += evt.cuda_time_total / 1e3
        bmm_pred_tot += pred
    elif evt.name == 'autograd::engine::evaluate_function: BmmBackward0':
        inputs = evt.cpu_children[0].input_shapes
        bmm_evt = evt.cpu_children[0].cpu_children[1]
        assert(bmm_evt.name == 'aten::bmm')
        inputs = bmm_evt.input_shapes
        batch_size, l, m, n = inputs[0][0], inputs[0][1], inputs[0][2], inputs[1][2]
        pred = bmm_pred.predict([batch_size, l, m, n, 1, use_fp16])
        bmm_tot += evt.cuda_time_total / 1e3
        bmm_pred_tot += pred
    elif evt.name.startswith('Optimizer'):
        # print(evt.name, evt.cpu_children)
        # go through its children
        pred = 0
        for child in evt.cpu_children:
            inputs = child.input_shapes
            size_sum = sum([np.prod(i) for i in inputs])
            pred += size_sum * perfpred.trace.UNARY_COEFF 
        print("OPTIMIZER:", evt.cuda_time_total / 1e3, pred)
    elif evt.name.startswith('autograd::engine::evaluate_function:'):
        inputs = evt.cpu_children[0].input_shapes 
        size_sum = sum([np.prod(i) for i in inputs]) 
        pred = size_sum * perfpred.trace.UNARY_COEFF 
        misc_pred_tot += pred
        misc_tot += evt.cuda_time_total / 1e3
    else:
        inputs = evt.input_shapes
        size_sum = sum([np.prod(i) for i in inputs]) 
        pred = size_sum * perfpred.trace.UNARY_COEFF
        misc_pred_tot += pred
        misc_tot += evt.cuda_time_total / 1e3
    pred_tot += pred

print(root.cpu_time_total/1e3, pred_tot,)
print('misc:', misc_tot, misc_pred_tot)
print('linear:', linear_tot, linear_pred_tot)
print('bmm:', bmm_tot, bmm_pred_tot)
torch.cuda.synchronize()