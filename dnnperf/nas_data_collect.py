import os
import subprocess
import itertools

import tqdm, random

model_range = range(6_500)

batch_size_range = range(16, 129)
image_size_range = range(32, 257, 32) 

for i in tqdm.trange(10000):
    model = random.choice(model_range)
    batch_size = random.choice(batch_size_range)
    image_size = random.choice(image_size_range)
    ret = subprocess.run(args=f'python nas_worker.py {model} {batch_size} {image_size}', shell=True, capture_output=True)
    if ret.returncode == 0:
        mem = float(ret.stdout.split() [-1])
        print(f"{model}, {batch_size}, {image_size}, {mem}")