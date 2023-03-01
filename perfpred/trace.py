import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import re

from .measure import measure_unary_elementwise
from .predictor import Conv2DPredictor, LinearPredictor, MaxPoolingPredictor, BatchNormPredictor

UNARY_COEFF = 1.50332785e-08 
UNARY_BIAS = 0
BINARY_COEFF = UNARY_COEFF * 1.5

def measure_simple_op():
    global UNARY_COEFF, BINARY_COEFF, UNARY_BIAS
    x_range = np.arange(200_000, 11_000_000, 200_000)
    y = []
    for n in x_range:
        dur = measure_unary_elementwise(n, op=F.relu)
        y.append(dur)
    A = np.vstack([x_range, np.ones(len(x_range))]).T
    ret, _, _, _ = np.linalg.lstsq(A, np.array(y), rcond=-1)
    UNARY_COEFF = ret[0]
    UNARY_BIAS = ret[1]
    BINARY_COEFF = 1.5 * UNARY_COEFF 
    print(UNARY_COEFF, BINARY_COEFF)

UNARY_COEFF = {
    "2080ti":1.463674020617832e-08
}

if __name__ == "__main__":
    print("Measuring Memory Bandwidth...")
    measure_simple_op()

def profile_model(func, nitr=20, device='cuda'):
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        schedule= torch.profiler.schedule(
            wait=1,
            warmup=5,
            active=nitr,
            repeat=1),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        for _ in range(nitr + 7):
            func()
            profiler.step()
        torch.cuda.synchronize(device)
    return profiler.profiler.function_events

TRACE_MODULE = (
    nn.Conv2d,
    nn.MaxPool2d,
    nn.Linear,
    nn.BatchNorm2d, 
)

class Tracer(object):
    def forward_hook(self, record):
        def fun(module, input, output):
            if not type(module) in TRACE_MODULE:
                return
            if len(list(module.children())) == 0:
                record.append(
                    (1, module, [None if i is None else i.shape for i in input], True)
                )
        return fun
    
    def backward_hook(self, record):
        def fun(module, input, output):
            if not type(module) in TRACE_MODULE:
                return
            dx = True
            has_grad = False
            for grad_output in output:
                if not torch.all(grad_output == 0):
                    has_grad = True
                    break
            if not has_grad:
                return
            if isinstance(module, nn.Conv2d):
                if input[0] is None:
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
        # nn.ReLU : ('autograd::engine::evaluate_function: ReluBackward0', 'aten::relu'),
        nn.Linear : ('autograd::engine::evaluate_function: (MmBackward0)|(AddmmBackward0)', 'aten::linear'),
        # nn.Flatten : ('autograd::engine::evaluate_function: ReshapeAliasBackward0', 'aten::flatten'),
        # nn.CrossEntropyLoss : (
        #     ('autograd::engine::evaluate_function: NllLossBackward0',
        #      'autograd::engine::evaluate_function: LogSoftmaxBackward0'), 
        #     'aten::cross_entropy_loss'),
        nn.BatchNorm2d : ('autograd::engine::evaluate_function: CudnnBatchNormBackward0', 'aten::batch_norm'),
        # nn.AdaptiveAvgPool2d: ('autograd::engine::evaluate_function: MeanBackward1', 'aten::adaptive_avg_pool2d'),
        # nn.Dropout : ("aten::native_dropout_backward", "aten::dropout")
    }

    def match_trace_and_events(self, trace, events, verbose=0):
        # clean the trace, remove unnecessary
        acc_grad = []
        optim_dur = []
        step = 0
        step_time = []
        all_kernel_time = []
        for idx, event in enumerate(events):
            if event.name.startswith('ProfilerStep'):
                break
    
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
            if ptr < len(trace):
                module = trace[ptr][1]
                is_forward = trace[ptr][0]
                while matching_names is None and ptr < len(trace):
                    for t, names in self.op_name_mapping.items():
                        if isinstance(module, t):
                            matching_names = names[is_forward]
                    assert matching_names is not None, module
                is_tuple = isinstance(matching_names, tuple)
                matching_name = matching_names[tuple_ptr] if is_tuple else matching_names
                if re.match(matching_name, c.name) is not None:
                    kernel_time = self._get_children_kernel_time(c, marked_kernel)
                    setattr(c, "module", module)
                    setattr(c, "is_forward", is_forward)
                    setattr(c, "input_shapes", trace[ptr][2])
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

        optim_dur.append(optim_t)
        step += 1
        trace_with_dur = []
        dur_counted = 0
        # for item, dur in zip(new_trace, dur_dict):
        #     trace_with_dur.append(item + (np.mean(dur), ))
        #     dur_counted += np.mean(dur)
        conv_time = 0
        linear_time = 0
        pool_time = 0
        bn_time = 0
        relu_time = 0

        # for debugging
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

        acc_grad = np.sum(np.mean(acc_grad, axis=1))
        optim_dur = np.mean(optim_dur)

        if verbose >= 1:
            print("Tracing:", conv_time/1e3, linear_time/1e3, pool_time/1e3, bn_time/1e3, relu_time/1e3, acc_grad/1e3 + optim_dur/1e3)

class Predictor(object):
    def __init__(self, target):
        self.target = target
        self._load_models()
        self.UNARY_COEFF = UNARY_COEFF[target]
     
    def _load_models(self):
        target = self.target
        self.conv_pred = Conv2DPredictor()
        self.conv_pred.load_model(f"./model/{target}/predictor_model_conv2d.th")

        self.linear_pred = LinearPredictor()
        self.linear_pred.load_model(f"./model/{target}/predictor_model_linear.th")

        self.maxpool_pred = MaxPoolingPredictor()
        self.maxpool_pred.load_model(f"./model/{target}/predictor_model_maxpool.th")

        self.batchnorm_pred = BatchNormPredictor()
        self.batchnorm_pred.load_model(f"./model/{target}/predictor_model_batchnorm.th")

    def _mark(self, visited, event):
        visited.add(event)
        for child in event.cpu_children:
            self._mark(visited, child)

    def predict_using_trace(self, trace, events, use_fp16=False, verbose=0):
        tot_time = 0
        conv_time = 0
        linear_time = 0
        pool_time = 0
        bn_time = 0
        relu_time = 0
        dur_list = []
        visited = set()
        for idx, event in enumerate(events):
            if event.name.startswith('ProfilerStep'):
                break
        children = Tracer._get_all_children(events, event)
        for event in children:
            # pred = None
            # print(event)
            if hasattr(event, 'module'):
                module = event.module
                is_forward = event.is_forward
                input_shapes = event.input_shapes
                # print(module, is_forward)
                self._mark(visited, event)

                if isinstance(module, nn.Conv2d):
                    # if is_forward:
                    input_shape = input_shapes[0]
                    if input_shape == None:
                        for f, m, shape, _ in trace:
                            if m == module and f:
                                input_shape = shape[0]
                    # else:
                        # input_shape = event.input_shapes[0]
                    pred = self.conv_pred.predict(
                        [0, 
                         input_shape[0], 
                         input_shape[2], 
                         input_shape[1], 
                         module.out_channels, 
                         module.kernel_size[0], 
                         module.stride[0], 
                         module.padding[0], 
                         is_forward,
                         use_fp16]
                    ) 
                    # print(pred)

                    # if module.bias is not None:
                    #     bias_pred = self.UNARY_COEFF * (input_shape[0] * ((input_shape[2] / module.stride[0]) ** 2) * module.out_channels)
                    #     if use_fp16:
                    #         bias_pred /= 2
                    #     pred += bias_pred
                    conv_time += pred
                    tot_time += pred
                if isinstance(module, nn.Linear):
                    input_shape = input_shapes[0]
                    pred = self.linear_pred.predict(
                        [module.bias is not None, input_shape[0], input_shape[1], module.out_features, is_forward, use_fp16]
                    )
                    # if not dx:
                        # pred /= 2
                    # if module.bias is not None:
                        # bias_pred = UNARY_COEFF * input_shape[0] * module.out_features
                        # if use_fp16:
                        #    bias_pred /= 2 
                    linear_time += pred
                    tot_time += pred
                if isinstance(module, nn.MaxPool2d):
                    input_shape = event.cpu_children[0].input_shapes[0]
                    pred = self.maxpool_pred.predict(
                        [input_shape[0], module.kernel_size, input_shape[2], input_shape[1], module.stride, is_forward, use_fp16]
                    )
                    pool_time += pred
                    tot_time += pred
                if isinstance(module, nn.BatchNorm2d):
                    # if is_forward:
                    input_shape = event.cpu_children[0].input_shapes[0]
                    # print(input_shape)
                    # else:
                        # input_shape = event.input_shapes[0]
                    pred = self.batchnorm_pred.predict(
                        [input_shape[0], input_shape[2], input_shape[1], is_forward, use_fp16]
                    )
                    tot_time += pred
                    bn_time += pred
                # print(event.name, module, pred)
            elif not event in visited:
                if len(event.kernels) > 0 and len(event.input_shapes) > 0:
                    input_size = sum([np.prod(s) for s in event.input_shapes])
                    pred = input_size * self.UNARY_COEFF
                    tot_time += pred
                    # print(event.name, event.cuda_time, pred)
            # dur_list.append(pred)
        
        if verbose >= 1:
            print("Predict:", conv_time, linear_time, pool_time, bn_time, relu_time)
        
        return tot_time, dur_list

    def predict(self, model, trace_func, use_fp16=False, verbose=0, dry_run=5):
        # dry run
        for _ in range(dry_run):
            trace_func()
        torch.cuda.synchronize()
        tracer = Tracer()
        trace = tracer.trace(trace_func)
        events = profile_model(trace_func)
        tracer.match_trace_and_events(trace, events, verbose=verbose)
        pred = self.predict_using_trace(trace, events, use_fp16, verbose)
        return pred

    def predict_cpu(self, ):
        pass

    def _load_cpu_map(self):
        pass