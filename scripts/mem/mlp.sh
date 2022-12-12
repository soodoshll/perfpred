#!bin/bash
cmd="python scripts/mem/mlp.py --nlayer 200 --ndim 2048"
if [ "$FAKE_ALLOC" == 1 ]
then
    echo "Using fake allocator"
    source ./faketorch/bin/activate
    cmd="env LD_PRELOAD=./fake_libcudart.so "$cmd
fi

for bs in {1024..10240..512}
do
    ret=$($cmd --batch_size $bs 2>/dev/null)
    echo $bs, $ret
    sleep 1
done