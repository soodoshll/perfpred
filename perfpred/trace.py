import torch
from torch import nn
import numpy as np
from vgg import build_vgg_model
from predictor import Conv2DPredictor, LinearPredictor, MaxPoolingPredictor, BatchNormPredictor
import torchvision
from resnet import ResNet18
import re
import argparse

from utils import timing 

device = torch.device('cuda')

UNARY_COEFF = 1.50332785e-08 
BINARY_COEFF = UNARY_COEFF * 1.5

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=int, default=0)
parser.add_argument("--model", choices=['resnet50', 'vgg'], default='vgg')
args = parser.parse_args()

def profile_model(func, nitr=20):
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        schedule= torch.profiler.schedule(
            wait=1,
            warmup=5,
            active=nitr,
            repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        # with_modules=True,
        record_shapes=True,
    ) as profiler:
        for _ in range(nitr + 7):
            func()
            profiler.step()
        torch.cuda.synchronize(device)
    return profiler.profiler.function_events

class Tracer(object):
    def forward_hook(self, record):
        def fun(module, input, output):
            # print(module)
            if len(list(module.children())) == 0:
                record.append(
                    (1, module, [None if i is None else i.shape for i in input], True)
                )
        return fun
    
    def backward_hook(self, record):
        def fun(module, input, output):
            # if isinstance(module, nn.Conv2d):
            dx = True
            if isinstance(module, nn.Conv2d):
                # print(len(input), len(output))
                # print(input[0].shape)
                # print(module)
                # print(input[0].shape)
                if input[0] is None:
                    # print("first layer")
                    dx = False
            if len(list(module.children())) == 0:
                record.append(
                    (0, module, [None if i is None else i.shape for i in input], dx)
                )
        return fun
    
    def trace(self, func):
        record = []
        forward_handle = nn.modules.module.register_module_forward_hook(self.forward_hook(record))
        backward_handle = nn.modules.module.register_module_full_backward_hook(self.backward_hook(record))
        func()
        forward_handle.remove()
        backward_handle.remove()
        return record
    
    @staticmethod
    def _get_all_children(events, root):
        ret = []
        for evt in events:
            if evt.time_range.start >= root.time_range.start and evt.time_range.end <= root.time_range.end:
                ret.append(evt)
        return ret


    @staticmethod
    def _get_children_kernel_time(event, marked_kernel=None):
        kernel_time = sum([kernel.duration for kernel in event.kernels])
        if marked_kernel is not None:
            marked_kernel.update(event.kernels)
        for child in event.cpu_children:
            kernel_time += Tracer._get_children_kernel_time(child, marked_kernel)
        return kernel_time

    @staticmethod
    def _find_parent_with_shapes(evt):
        if evt.cpu_parent is None or len(evt.input_shapes) > 0:
            return evt
        return Tracer._find_parent_with_shapes(evt)

    op_name_mapping = {
        nn.Conv2d : ('autograd::engine::evaluate_function: ConvolutionBackward0', 'aten::conv2d'),
        nn.MaxPool2d : ('autograd::engine::evaluate_function: MaxPool2DWithIndicesBackward0', 'aten::max_pool2d'),
        nn.ReLU : ('autograd::engine::evaluate_function: ReluBackward0', 'aten::relu'),
        nn.Linear : ('autograd::engine::evaluate_function: (MmBackward0)|(AddmmBackward0)', 'aten::linear'),
        nn.Flatten : ('autograd::engine::evaluate_function: ReshapeAliasBackward0', 'aten::flatten'),
        nn.CrossEntropyLoss : (
            ('autograd::engine::evaluate_function: NllLossBackward0',
             'autograd::engine::evaluate_function: LogSoftmaxBackward0'), 
            'aten::cross_entropy_loss'),
        nn.BatchNorm2d : ('autograd::engine::evaluate_function: CudnnBatchNormBackward0', 'aten::batch_norm'),
        nn.AdaptiveAvgPool2d: ('autograd::engine::evaluate_function: MeanBackward1', 'aten::adaptive_avg_pool2d')
    }


    def match_trace_and_events(self, trace, events):
        # clean the trace, remove unnecessary
        new_trace = []
        for item in trace:
            if not isinstance(item[1], nn.Sequential):
                new_trace.append(item)
        dur_dict = [[] for i in range(len(new_trace))]
        acc_grad = []
        optim_dur = []
        step = 0
        step_time = []
        all_kernel_time = []
        for idx, event in enumerate(events):
            if event.name.startswith('ProfilerStep'):
                # print(event.name, event.self_cpu_time_total)
                step_kernel_time = 0
                marked_kernel = set()
                step_time.append(event.cpu_time_total)
                children = self._get_all_children(events, event)
                ptr = 0
                acc_grad_ptr = 0
                matching_names = None
                optim_t = 0

                tuple_ptr = 0
                for c in children:
                    step_kernel_time += sum([k.duration for k in c.kernels])
                    if ptr < len(new_trace):
                        module = new_trace[ptr][1]
                        is_forward = new_trace[ptr][0]
                        while matching_names is None and ptr < len(new_trace):
                            for t, names in self.op_name_mapping.items():
                                if isinstance(module, t):
                                    matching_names = names[is_forward]
                            assert matching_names is not None, module
                            # if matching_name is None:
                            #     ptr += 1
                            #     module = new_trace[ptr][1]
                            #     is_forward = new_trace[ptr][0]
                        is_tuple = isinstance(matching_names, tuple)
                        matching_name = matching_names[tuple_ptr] if is_tuple else matching_names
                        if re.match(matching_name, c.name) is not None:
                            kernel_time = self._get_children_kernel_time(c, marked_kernel)
                            if tuple_ptr > 0:
                                dur_dict[ptr][-1] += kernel_time
                            else:
                                dur_dict[ptr].append(kernel_time)
                            if (is_tuple): 
                                tuple_ptr += 1
                            if (not is_tuple or tuple_ptr == len(matching_names)):
                                ptr += 1
                                tuple_ptr = 0
                                matching_names = None
                    if c.name.startswith('Optimizer.step'):
                        optim_t += self._get_children_kernel_time(c, marked_kernel)
                    if c.name == 'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad':
                        kernel_time = self._get_children_kernel_time(c, marked_kernel)
                        if step == 0:
                            acc_grad.append([kernel_time])
                        else:
                            acc_grad[acc_grad_ptr].append(kernel_time)
                            acc_grad_ptr += 1
                
                all_kernel_time.append(step_kernel_time)
                
                unmarked_event = []
                for c in children:
                    for kernel in c.kernels:
                        if not kernel in marked_kernel:
                            unmarked_event.append(self._find_parent_with_shapes(c))
                            break
                # print(unmarked_event)

                optim_dur.append(optim_t)
                step += 1
        # print(dur_dict)
        trace_with_dur = []
        dur_counted = 0
        for item, dur in zip(new_trace, dur_dict):
            trace_with_dur.append(item + (np.mean(dur), ))
            dur_counted += np.mean(dur)
            # print(np.mean(dur))
        # print(trace_with_dur)
        conv_time = 0
        linear_time = 0
        pool_time = 0
        bn_time = 0
        relu_time = 0
        for is_forward, module, _, _, dur in trace_with_dur:
            if isinstance(module, nn.Conv2d):
                conv_time += dur
            if isinstance(module, nn.Linear):
                linear_time += dur
            if isinstance(module, nn.MaxPool2d):
                pool_time += dur
            if isinstance(module, nn.BatchNorm2d):
                bn_time += dur
            if isinstance(module, nn.ReLU):
                relu_time += dur
        # print
        acc_grad = np.sum(np.mean(acc_grad, axis=1))
        optim_dur = np.mean(optim_dur)
        # print(acc_grad)
        # print(dur_counted, acc_grad, optim_dur)
        # print((dur_counted + acc_grad + optim_dur) / 1e3, np.mean(step_time) / 1e3)

        if args.verbose >= 1:
            print("Tracing:", conv_time/1e3, linear_time/1e3, pool_time/1e3, bn_time/1e3, relu_time/1e3, acc_grad/1e3 + optim_dur/1e3)
        # return trace_with_dur, acc_grad, optim_dur, np.mean(step_time) / 1e3
        return np.mean(step_time) / 1e3, np.mean(all_kernel_time)/1e3 , unmarked_event, trace_with_dur
        # return conv_time / 1e3

# model = ResNet18()

def basicblock_revised_forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = self.relu(out)

    return out

def bottleneck_revised_forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = self.relu(out)

    return out

torchvision.models.resnet.BasicBlock.forward = basicblock_revised_forward
torchvision.models.resnet.Bottleneck.forward = bottleneck_revised_forward

if args.model == 'vgg':
    model = build_vgg_model(bias=False)
elif args.model == 'resnet50':
    model = torchvision.models.resnet50()
else:
    raise RuntimeError("not supported")

# print(model)
def _change_inplace_to_false(module):
    if hasattr(module, 'inplace'):
        # print(module)
        module.inplace = False

model.apply(_change_inplace_to_false)

model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

conv_pred = Conv2DPredictor(True)
conv_pred.load_model("./model/predictor_model_conv2d.th")

linear_pred = LinearPredictor()
linear_pred.load_model("./model/predictor_model_linear.th")

maxpool_pred = MaxPoolingPredictor()
maxpool_pred.load_model("./model/predictor_model_maxpool.th")

batchnorm_pred = BatchNormPredictor()
batchnorm_pred.load_model("./model/predictor_model_batchnorm.th")

use_fp16 = True

def predict_using_trace(model, trace):
    tot_time = 0
    conv_time = 0
    linear_time = 0
    pool_time = 0
    bn_time = 0
    relu_time = 0
    dur_list = []
    for is_forward, module, input_shapes, dx in trace:
        pred = None
        if isinstance(module, nn.Conv2d):
            input_shape = input_shapes[0]
            if input_shape == None:
                for f, m, shape, _ in trace:
                    if m == module and f:
                        input_shape = shape[0]
            pred = conv_pred.predict(
                [0, input_shape[0], input_shape[2], input_shape[1], module.out_channels, module.kernel_size[0], module.stride[0], module.padding[0], is_forward, use_fp16]
            ) 
            if not dx:
                pred /= 2
            # print([0, input_shape[0], input_shape[2], input_shape[1], module.out_channels, module.kernel_size[0], module.stride[0], module.padding[0], is_forward, use_fp16])
            if module.bias is not None:
                bias_pred = 1.50332785e-08 * (input_shape[0] * ((input_shape[2] / module.stride[0]) ** 2) * module.out_channels)
                if use_fp16:
                    bias_pred /= 2
                pred += bias_pred
            conv_time += pred
            tot_time += pred
        if isinstance(module, nn.Linear):
            input_shape = input_shapes[0]
            pred = linear_pred.predict(
                [module.bias is not None, input_shape[0], input_shape[1], module.out_features, is_forward, use_fp16]
            )
            if not dx:
                pred /= 2
            # if module.bias is not None:
                # bias_pred = UNARY_COEFF * input_shape[0] * module.out_features
                # if use_fp16:
                #    bias_pred /= 2 
            linear_time += pred
            tot_time += pred
        if isinstance(module, nn.MaxPool2d):
            input_shape = input_shapes[0]
            pred = maxpool_pred.predict(
                [input_shape[0], module.kernel_size, input_shape[2], input_shape[1], module.stride, is_forward, use_fp16]
            )
            pool_time += pred
            tot_time += pred
        if isinstance(module, nn.ReLU):
            input_shape = input_shapes[0]
            input_size = np.prod(input_shape)
            if is_forward:
                pred = UNARY_COEFF * input_size
            else:
                pred = BINARY_COEFF * input_size
            if use_fp16:
                pred /= 2
            tot_time += pred
            relu_time += pred
        if isinstance(module, nn.BatchNorm2d):
            input_shape = input_shapes[0]
            pred = batchnorm_pred.predict(
                [input_shape[0], input_shape[2], input_shape[1], is_forward, use_fp16]
            )
            tot_time += pred
            bn_time += pred
        dur_list.append(pred)
    
    # optimizer
    param_size = 0
    for param in model.parameters():
        param_size += np.prod(param.size())
    optim_time = (1.50332785e-08 + 2.19720769e-08) * param_size 
    if use_fp16:
        optim_time /= 2
    tot_time += optim_time

    if args.verbose >= 1:
        print("Predict:", conv_time, linear_time, pool_time, bn_time, relu_time, optim_time)
    return tot_time, dur_list
    # return conv_time

scaler = torch.cuda.amp.GradScaler()

for batch_size in [8 ,16 ,32]:
    for use_fp16 in [False, True]:
        if use_fp16:
            scaler = torch.cuda.amp.GradScaler() 
        tracer = Tracer()
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


        dur_measure = timing(trace_func, 3, 3)

        trace = tracer.trace(trace_func)
        pred, pred_dur = predict_using_trace(model, trace)

        events = profile_model(trace_func)
        truth, truth_kernel_time, unmarked_events, trace_with_dur = tracer.match_trace_and_events(trace, events)


        unmarked_tot = 0
        unmarked_pred = 0
        for evt in unmarked_events:
            t = BINARY_COEFF * np.prod(evt.input_shapes[0])
            if use_fp16:
                t /= 2
            unmarked_pred += t
            pred += t
            unmarked_tot += evt.cuda_time_total
        
        # if verbo
        # print(unmarked_pred, unmarked_tot / 1e3)
            # print(evt.name, evt.cuda_time_total, t * 1e3)

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
        del inputs, labels, trace, events, tracer