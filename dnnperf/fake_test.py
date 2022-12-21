ID_FILE = "dnnperf/dataset_index.npz"
DATA_FILE = "dnnperf/nas_mem_measure.txt"

import numpy as np
import subprocess
id_dict = np.load(ID_FILE)
test_id = id_dict['test_id']

with open(DATA_FILE) as f:
    data = []
    cnt = 0
    for line in f:
        if line.strip() == '' or line.startswith('b'):
            continue
        tokens = line.split(',')
        model_id = int(tokens[0])
        batch_size = int(tokens[1])
        image_size = int(tokens[2])
        mem = float(tokens[3])
        if cnt in test_id and cnt > 1063:
            data.append([model_id, batch_size, image_size, mem])
        cnt += 1

for model_id, batch_size, image_size, mem in data:
    # p = subprocess.run(f"LD_PRELOAD=./fake_libcudart.so python dnnperf/fake_runner.py {model_id} {batch_size} {image_size}", shell=True, capture_output=True)
    # ret = float(p.stdout.splitlines()[-1])
    # print(ret, mem)
    print(f"LD_PRELOAD=./fake_libcudart.so python dnnperf/fake_runner.py {model_id} {batch_size} {image_size}")