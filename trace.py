from re import L
import torch
from torch import nn
import numpy as np
from vgg import build_vgg_model

device = torch.device('cuda')

def profile_model(func, nitr=5):
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        schedule= torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=nitr,
            repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        # with_modules=True,
        record_shapes=True,
    ) as profiler:
        for _ in range(nitr + 3):
            func()
            profiler.step()
        torch.cuda.synchronize(device)
    return profiler.profiler.function_events

class Tracer(object):
    def forward_hook(self, record):
        def fun(module, input, output):
            record.append(
                (1, module, [None if i is None else i.shape for i in input])
            )
        return fun
    
    def backward_hook(self, record):
        def fun(module, input, output):
            record.append(
                (0, module, [None if i is None else i.shape for i in input])
            )
        return fun
    
    def trace(self, func):
        record = []
        forward_handle = nn.modules.module.register_module_forward_hook(self.forward_hook(record))
        backward_handle = nn.modules.module.register_module_backward_hook(self.backward_hook(record))
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
    def _get_children_kernel_time(event):
        kernel_time = sum([kernel.duration for kernel in event.kernels])
        for child in event.cpu_children:
            kernel_time += Tracer._get_children_kernel_time(child)
        return kernel_time

    op_name_mapping = {
        nn.Conv2d : ('autograd::engine::evaluate_function: ConvolutionBackward0', 'aten::conv2d'),
        nn.MaxPool2d : ('autograd::engine::evaluate_function: MaxPool2DWithIndicesBackward0', 'aten::max_pool2d'),
        nn.ReLU : ('autograd::engine::evaluate_function: ReluBackward0', 'aten::relu'),
        nn.Linear : ('autograd::engine::evaluate_function: MmBackward0', 'aten::linear'),
        nn.Flatten : ('autograd::engine::evaluate_function: ReshapeAliasBackward0', 'aten::flatten')
    }

    def match_trace_and_events(self, trace, events):
        # clean the trace, remove unnecessary
        new_trace = []
        for item in trace:
            if not isinstance(item[1], nn.Sequential):
                new_trace.append(item)
        dur_dict = [[] for i in range(len(new_trace))]
        acc_grad = []
        first_step = True
        for event in events:
            if event.name.startswith('ProfilerStep'):
                print(event.name)
                children = self._get_all_children(events, event)
                ptr = 0
                acc_grad_ptr = 0
                matching_name = None
                for c in children:
                    # if c.name.find('autograd::engine') >= 0:
                        # print(c.name, c.cpu_parent, self._get_children_kernel_time(c))
                    if ptr < len(new_trace):
                        module = new_trace[ptr][1]
                        is_forward = new_trace[ptr][0]
                        while matching_name is None:
                            for t, names in self.op_name_mapping.items():
                                if isinstance(module, t):
                                    matching_name = names[is_forward]
                            assert matching_name is not None
                            # print(matching_name)
                        if c.name == matching_name:
                            kernel_time = self._get_children_kernel_time(c) / 1e3
                            dur_dict[ptr].append(kernel_time)   
                            ptr += 1
                            matching_name = None
                    if c.name == 'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad':
                        kernel_time = self._get_children_kernel_time(c) / 1e3
                        if first_step:
                            acc_grad.append([kernel_time])
                        else:
                            acc_grad[acc_grad_ptr].append(kernel_time)
                            acc_grad_ptr += 1
                first_step = False
        # print(dur_dict)
        trace_with_dur = []
        for item, dur in zip(new_trace, dur_dict):
            trace_with_dur.append(item + (np.mean(dur), ))
        print(trace_with_dur)
        return dur_dict, acc_grad



tracer = Tracer()

input = torch.rand([32, 3, 224, 224], device=device)
model = build_vgg_model(bias=False)
model.to(device)

def trace_func():
    out = model(input)
    out = out.sum()
    out.backward()
    torch.cuda.synchronize()

trace = tracer.trace(trace_func)
events = profile_model(trace_func)

tracer.match_trace_and_events(trace, events)
