#!/bin/bash

source torch_env/bin/activate
cmd="python scripts/time/batchsize_cpu_gpu.py "

model=densenet121
for bs in {1..64}; do
    ret=$($cmd --model $model $bs 2> /dev/null)
    echo $model, $bs, $ret
done
