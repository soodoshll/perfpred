import torch
import transformers
from transformers import BertConfig, BertModel, GPT2Model, GPT2Config, BertForPreTraining, BertForSequenceClassification
import os
from ctypes import cdll
from apex import amp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--amp', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=512)
args = parser.parse_args()

# os.environ['LD_PRELOAD'] = './fake_libcudart.so'

# cdll.LoadLibrary('./fake_libcudart.so')

use_fake_alloc = os.environ.get("LD_PRELOAD", None) == "./fake_libcudart.so"
print("Using fake allocator:", use_fake_alloc)
if use_fake_alloc:
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    import fake_alloc

def empty_init(*args, **kwargs):
    # print("you're fucked up, initializer")
    return

initializers = [
    "xavier_normal", 
    "xavier_uniform",
    "normal",
    "trunc_normal",
    "uniform",
    "zeros",
    "eye",
    "constant",
    "ones",
    "dirac",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal"
]

for init in initializers:
    exec(f"torch.nn.init.{init} = empty_init")
    exec(f"torch.nn.init.{init}_ = empty_init")

device = torch.device('cuda')
torch.cuda.empty_cache()
# configuration = GPT2Config(n_layer=24, n_embd=1024, n_head=16)
# configuration = GPT2Config()
# model = GPT2Model(configuration)

# configuration = BertConfig()
# model = BertModel(configuration)

# transformers.modeling_utils._init_weights = False
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.zeros((args.batch_size, args.seq_len), device=device, dtype=torch.int32)
# fake_alloc.init_max_mem()
print(torch.cuda.max_memory_allocated() / 1e6)
model, optim = amp.initialize(model, optim, opt_level="O" + str(args.amp), verbosity=0)

parameter_size = 0
for p in model.parameters():
    parameter_size += p.data.element_size() * p.data.nelement()

# print("param:", parameter_size)

class SavedTensorRecord(object):
    saved_tensor_size = 0

    @property
    def pack_hook(self):
        def foo(x):
            self.saved_tensor_size += x.element_size() * x.nelement()
            return x
        return foo

    @property
    def unpack_hook(self):
        def foo(x):
            return x
        return foo
    
    def reset(self):
        self.saved_tensor_size = 0

saved_tensor_record = SavedTensorRecord()
if use_fake_alloc:
    fake_alloc.init_max_mem()

for i in range(5):
    saved_tensor_record.reset()

    with torch.autograd.graph.saved_tensors_hooks(saved_tensor_record.pack_hook, saved_tensor_record.unpack_hook):
    # with torch.no_grad():
        optim.zero_grad()
        out = model(x)
        # print(torch.cuda.memory_allocated())
        # print(out.keys())
        # print(out)
        loss = out['logits'].sum()
    # loss = out['pooler_output'].sum() + out['last_hidden_state'].sum()
        # loss = out['prediction_logits'].sum()
        # loss = out.sum()
        with amp.scale_loss(loss, optim, delay_overflow_check=True) as scaled_loss:
            scaled_loss.backward()
        optim.step()
        del loss, out
# print("activation:", saved_tensor_record.saved_tensor_size)

def tensor_size(x):
    if x is None:
        return 0
    return x.element_size() * x.nelement()

state_dict = optim.state_dict()
state_total = 0
for _, item in state_dict['state'].items():
    # print()
    state_total += tensor_size(item['exp_avg']) + tensor_size(item['exp_avg_sq'])
# print("states:", state_total)
# out.backward()

gradient_total = 0
for p in model.parameters():
    gradient_total += tensor_size(p.grad)

# print("grad:", gradient_total)

# print(saved_tensor_record.saved_tensor_size + state_total + parameter_size + gradient_total)


if use_fake_alloc:
    peak_mem = fake_alloc.max_mem_allocated()
else:
    peak_mem = torch.cuda.max_memory_allocated()

print(f"{(parameter_size) / 1e9 : .2f}, {(state_total) / 1e9 :.2f}, {gradient_total / 1e9 : .2f}, {saved_tensor_record.saved_tensor_size / 1e9 :.2f}, {peak_mem / 1e9 :.2f}")