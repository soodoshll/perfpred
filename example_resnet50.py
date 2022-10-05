import torch
import torchvision
from perfpred.vgg import build_vgg_model
import torchdynamo

import argparse
from perfpred.utils import timing
torchdynamo.config.verbose=True
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

device = 'cuda'

model = build_vgg_model()
# model = torchvision.models.resnet50()
model.to(device)

optim = torch.optim.SGD(model.parameters(), lr=1e-3)
x = torch.zeros((args.batch_size, 3, 224, 224), device=device, dtype=torch.float32)
model(x)

@torchdynamo.optimize("inductor")
def fw(x):
    return model(x)

def train_loop():
    optim.zero_grad()
    # with torch.no_grad():
    out = fw(x)
    loss = out.sum()
    loss.backward()
    optim.step()

dur = timing(train_loop)
torch.cuda.synchronize()
print(dur)
# p.export_chrome_trace("trace_transformer.json")