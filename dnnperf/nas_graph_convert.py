import torch
from trace import Graph
import tqdm
import pickle, time

import xautodl
from xautodl.models import get_cell_based_tiny_net
from nats_bench import create
import gc
device = 'cuda'

api = create(None, 'tss', fast_mode=True, verbose=False)

in_file = 'nas_mem_measure.txt'
out_file = 'nas_graphs.data'
cnt = 0
cnt_succ = 0
with open(in_file) as in_f, open(out_file, 'wb') as out_f:
    data = []
    for line in in_f:
        if line.strip() == '' or line.startswith('b'):
            continue
        tokens = line.split(',')
        cnt += 1
        model_id = int(tokens[0])
        batch_size = int(tokens[1])
        image_size = int(tokens[2])
        mem = float(tokens[3])

        config = api.get_net_config(model_id, 'ImageNet16-120')
        oom = False
        try:
            model = get_cell_based_tiny_net(config)
            # print(model)
            model.to(device)
            model.train()

            inputs = torch.rand((batch_size, 3, image_size, image_size), device=device)
            out = model(inputs)[0]
            g = Graph(out)
            g_data = g.encode()
            data.append(g_data + (mem,))
            del inputs, model, out, g, # g_data
            cnt_succ += 1
        except RuntimeError: # Out of memory
            oom = True
        print(f"[{cnt_succ}/{cnt}]")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    pickle.dump(data, out_f)
        
