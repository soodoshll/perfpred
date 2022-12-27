# new version of the tracer and predictor
# take transformer as an example

import torch
from torch.autograd import DeviceType

import transformers
from transformers import BertConfig, BertModel, GPT2Model, GPT2Config, BertForPreTraining, BertForSequenceClassification, GPT2ForSequenceClassification
import os
from ctypes import cdll
import argparse
from perfpred.utils import timing_cpu

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
parser.add_argument('--model', choices=['bert', 'gpt'], default='bert')
args = parser.parse_args()

device = 'cuda'

if args.model == 'bert':
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
elif args.model == 'gpt':
    model = GPT2ForSequenceClassification.from_pretrained('gpt2')
    model.config.pad_token_id = model.config.eos_token_id

model.to(device)
model.train()

optim = torch.optim.SGD(model.parameters(), lr=1e-3)
x = torch.zeros((args.batch_size, args.seq_len), device=device, dtype=torch.int32)
use_fp16 = args.amp

def lowest_level_func(evt):
    def _lowest_level_func(evt, l):
        if len(evt.cpu_children) == 1 and evt.cpu_children[0].name == 'cudaLaunchKernel':
            l.append(evt)
        elif len(evt.kernels) > 0:
            l.append(evt)
        else:
            for c in evt.cpu_children:
                _lowest_level_func(c, l)
    l = []
    _lowest_level_func(evt, l)
    return l

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

def _get_first_level_ops(trace, root):
    children = _get_all_children(trace, root)
    first_level_ops = []
    for evt in children:
        if evt.device_type == DeviceType.CPU and (evt.cpu_parent is None or evt.cpu_parent == root): # first level operators
            first_level_ops.append(evt)
    return first_level_ops

first_level_ops = _get_first_level_ops(trace, root)

pred_tot = 0
unknown_tot = 0

linear_tot_fw = 0
linear_pred_tot_fw = 0

linear_tot_bw = 0
linear_pred_tot_bw = 0

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
    elif evt.name == 'aten::linear' or evt.name=='aten::addmm':
        inputs = [i for i in evt.input_shapes if i != []]
        bias = len(inputs) == 3
        if bias:
            if len(inputs[0]) == 1:
                inputs = inputs[1:]
            else:
                inputs = inputs[:-1]
        # print(inputs)
        n =  np.prod(inputs[0][:-1])
        m = inputs[0][-1]
        if inputs[1][1] == m:
            k = inputs[1][0]
        else:
            k = inputs[1][1]
        # k = inputs[1][1] if len(inputs[1]) > 1 else 1
        pred = linear_pred.predict([bias, n, m, k, 1, use_fp16])
        # print(inputs, n, m, k, pred, evt.cuda_time_total / 1e3)
        linear_tot_fw += evt.cuda_time_total / 1e3
        linear_pred_tot_fw += pred
    elif evt.name == 'autograd::engine::evaluate_function: AddmmBackward0':
        children = evt.cpu_children
        mm1 = children[0].cpu_children[1]
        assert(mm1.name == 'aten::mm')
        inputs = mm1.input_shapes
        bias = True
        n, m, k = inputs[0][0], inputs[0][1], inputs[1][1]
        pred = linear_pred.predict([bias, n, m, k, 0, use_fp16])
        # print(inputs, n, m, k, pred, evt.cuda_time_total / 1e3)
        linear_tot_bw += evt.cuda_time_total / 1e3
        linear_pred_tot_bw += pred
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
        evt_pred_tot = 0
        for c in lowest_level_func(evt):
            inputs = evt.cpu_children[0].input_shapes 
            size_sum = sum([np.prod(i) for i in inputs])
            events = lowest_level_func(evt)
            pred = size_sum * perfpred.trace.UNARY_COEFF
            evt_pred_tot += pred
        # print(events)
        misc_pred_tot += evt_pred_tot
        pred = evt_pred_tot
        misc_tot += evt.cuda_time_total / 1e3
    else:
        evt_pred_tot = 0
        for c in lowest_level_func(evt): 
            inputs = evt.input_shapes
            size_sum = sum([np.prod(i) for i in inputs]) 
            pred = size_sum * perfpred.trace.UNARY_COEFF
            evt_pred_tot += pred
        misc_pred_tot += evt_pred_tot
        pred = evt_pred_tot
        misc_tot += evt.cuda_time_total / 1e3
    pred_tot += pred

print(pred_tot, root.cpu_time_total/1e3)
print('misc:', misc_pred_tot, misc_tot)
print('linear forward:', linear_pred_tot_fw, linear_tot_fw)
print('linear backward:', linear_pred_tot_bw, linear_tot_bw)
print('bmm:', bmm_pred_tot, bmm_tot)
torch.cuda.synchronize()