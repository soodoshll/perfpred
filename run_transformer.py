import torch
from transformers import BertConfig, BertModel, GPTJModel, GPTJConfig
import os

use_fake_alloc = os.environ.get("LD_PRELOAD", None) == "./fake_libcudart.so"
print("Using fake allocator:", use_fake_alloc)
if use_fake_alloc:
    import fake_alloc

device = torch.device('cuda')
configuration = GPTJConfig(n_positions=2048, n_embd=4096, n_layer=28, n_head=16)
model = GPTJModel(configuration)
# model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)

x = torch.zeros((1, 512), device=device, dtype=torch.int32)
out = model(x)
last_hidden_states = out.last_hidden_state
out = last_hidden_states.sum()
out.backward()
# out.backward()

if use_fake_alloc:
    print(fake_alloc.max_mem_allocated())
else:
    print(torch.cuda.max_memory_allocated())