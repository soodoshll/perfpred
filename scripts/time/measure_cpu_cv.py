import torch
import torchvision

import numpy as np
from perfpred.utils import profile_model, _get_first_level_ops, torch_vision_model_revise

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('command', choices=['measure', 'predict'])
parser.add_argument('--local', choices=['2070', '2080ti', '3090', 't4', 'v100'], default='2070')
parser.add_argument('--target', choices=['2070', '2080ti', '3090', 't4', 'v100'], default='2080ti')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--amp', action='store_true')

torch.backends.cudnn.benchmark = True

def _get_filename(device, amp):
    return f"./data/cpu_cnn_{device}_{amp}.data"

def _get_trainloop(model, device, amp, batch_size=1):
    image_size = 299 if model == 'inception_v3' else 224
    model = getattr(torchvision.models, model)()
    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    # print(image_size)
    inputs = torch.empty((batch_size, 3, image_size, image_size), device=device)
    labels = torch.zeros((batch_size, ), dtype=torch.int64, device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    def train_loop():
        optim.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
            out = model(inputs)
            if args.model == 'inception_v3':
                out = out[0]
            loss = loss_fn(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    return train_loop


def measure(args):
    device = 'cuda'
    train_loop = _get_trainloop(args.model, device, args.amp)
    events = profile_model(train_loop)
    evt_time_dict = {}

    start_time = []
    end_time = []

    def _add_event(evt):
        start_time.append(evt.time_range.start)
        end_time.append(evt.time_range.end)
        if evt.name in evt_time_dict:
            evt_time_dict[evt.name].append(evt.cpu_time_total)
        else:
            evt_time_dict[evt.name] = []

    train_loop()
    tot_time = []
    for root in events:
        if root.name.startswith("ProfilerStep*"):
            first_level_ops = _get_first_level_ops(events, root)
            tot_time.append(root.cpu_time_total)
            for op in first_level_ops:
                if op == root or op.name == 'cudaDeviceSynchronize':
                    continue
                if op.name.startswith('Optimizer.step'):
                    for c in op.cpu_children:
                        _add_event(c)
                elif op.name.startswith('Optimizer.zero_grad'):
                    for c in op.cpu_children:
                        _add_event(c)
                        # print(c.name)
                else:
                    _add_event(op)

    tot_gap = 0            
    for i in range(len(start_time) - 1):
        gap = start_time[i + 1] - end_time[i]
        tot_gap += gap

    out = {}
    for op, times in evt_time_dict.items():
        out[op] = np.mean(times)
        # print(op, out[op])

    out['GAP'] = tot_gap/1e3
    out['TOT'] = np.mean(tot_time)/1e3

    with open(_get_filename(args.local, args.amp), 'wb') as f:
        pickle.dump(out, f)

def predict(args):
    device = 'cuda'
    with open(_get_filename(args.local, args.amp), 'rb') as f:
        local_op_dict = pickle.load(f)
    with open(_get_filename(args.target, args.amp), 'rb') as f:
        target_op_dict = pickle.load(f)
    gap_ratio = target_op_dict['GAP'] / local_op_dict['GAP']
    tot_ratio = target_op_dict['TOT'] / local_op_dict['TOT']
    print("data loaded", gap_ratio, tot_ratio)
    train_loop = _get_trainloop(args.model, device, args.amp, args.batch_size)
    train_loop()

    events = profile_model(train_loop, dump_file="trace_cnn.json")
    start_time = []
    end_time = [] 
    tot_time = []
    op_dict = {}
    op_pred_dict = {}

    def _add_event(evt):
        if evt in target_op_dict:
            tot_time.append(target_op_dict[evt.name])
        else:
            tot_time.append(tot_ratio * evt.cpu_time_total)
        # print(evt.name, tot_time[-1])
        op_dict[evt.name] = op_dict.get(evt.name, 0) + evt.cpu_time_total
        op_pred_dict[evt.name] = op_pred_dict.get(evt.name, 0) + tot_time[-1] 
        start_time.append(evt.time_range.start)
        end_time.append(evt.time_range.end)

    for root in events:
        if root.name.startswith("ProfilerStep*"): 
            break

    for op in _get_first_level_ops(events, root):
        if op == root or op.name == 'cudaDeviceSynchronize':
            continue
        if op.name.startswith('Optimizer.step'):
            for c in op.cpu_children:
                _add_event(c)
        elif op.name.startswith('Optimizer.zero_grad'):
            for c in op.cpu_children:
                _add_event(c)
        else:
            _add_event(op)
    
    tot_gap = 0            
    for i in range(len(start_time) - 1):
        gap = start_time[i + 1] - end_time[i]
        tot_gap += gap
    # for op, t in op_dict.items():
        # print(op, t, op_pred_dict[op])
    # print(root.cpu_time_total/1e3)
    print(sum(tot_time)/1e3 + tot_gap * gap_ratio/1e3)
    
if __name__ == '__main__':
    args = parser.parse_args()
    if args.command == 'measure':
        measure(args)
    if args.command == 'predict':
        predict(args)
