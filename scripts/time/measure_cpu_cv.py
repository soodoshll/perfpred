import torch
import torchvision

import numpy as np

from perfpred.utils import profile_model, _get_first_level_ops, torch_vision_model_revise

torch_vision_model_revise()

device = 'cuda'
model = torchvision.models.resnet18()
model.to(device)

inputs = torch.empty((16, 3, 224, 224), device=device)
labels = torch.zeros((16, ), dtype=torch.int64, device=device)

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


for evt in events:
    if evt.name.startswith("ProfilerStep*"):
        first_level_ops = _get_first_level_ops(events, evt)
        for op in first_level_ops:
            if op.name in evt_time_dict:
                evt_time_dict[op.name].append(op.cpu_time_total)
            else:
                evt_time_dict[op.name] = []

for op, times in evt_time_dict.items():
   print(op, np.mean(times), np.std(times)) 