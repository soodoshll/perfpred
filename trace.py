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
    def _get_all_children(self, events, root):
        for evt in events:
            if event.

    def match_trace_and_events(self, trace, events):
        # clean the trace, remove unnecessary
        new_trace = []
        for item in trace:
            if not isinstance(item[1], nn.Sequential):
                new_trace.append(item)
        print(new_trace)
        # print(events)
        for event in events:
            # print(event.name)
            if event.name.startswith('ProfilerStep'):
                print(event.name)

tracer = Tracer()

input = torch.rand([1, 3, 32, 32], device=device)
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
