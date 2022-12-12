#!bin/bash
cmd="python scripts/mem/mlp.py --ndim 4096"
if [ "$FAKE_ALLOC" == 1 ]
then
    source ./faketorch/bin/activate
    preload_path=./fake_libcudart.so
fi

for bs in {1024..8192..512}
do
    env LD_PRELOAD=$preload_path $cmd --batch_size $bs 2>/dev/null;
    sleep 1
done