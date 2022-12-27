import torch
import torchvision

import numpy as np

from perfpred.utils import profile_model, _get_first_level_ops, torch_vision_model_revise

torch_vision_model_revise()

device = 'cuda'
model = torchvision.models.resnet18()
model.to(device)

inputs = torch.empty((1, 3, 224, 224), device=device)
labels = torch.zeros((1, ), dtype=torch.int64, device=device)

optim = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

def train_loop():
    optim.zero_grad()
    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()
    optim.step()

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

tot_time = []

for evt in events:
    if evt.name.startswith("ProfilerStep*"):
        first_level_ops = _get_first_level_ops(events, evt)
        tot_time.append(evt.cpu_time_total)
        for op in first_level_ops:
            if op == evt:
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


for op, times in evt_time_dict.items():
   print(op, np.mean(times), np.std(times)) 

print("GAP:", tot_gap/1e3)
print("TOT:", np.mean(tot_time)/1e3)