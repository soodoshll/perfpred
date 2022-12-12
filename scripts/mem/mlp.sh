#!bin/bash
cmd="python scripts/mem/mlp.py"
if [ "$FAKE_ALLOC" == 1 ]
then
    echo "Using fake allocator"
    source ./faketorch/bin/activate
    cmd="env LD_PRELOAD=./fake_libcudart.so "$cmd
fi

# change batch size
# echo "Change Batch Size.."
# for bs in {1024..10240..512}
# do
#     ret=$($cmd --nlayer 200 --ndim 2048 --batch_size $bs 2>/dev/null)
#     echo $bs, $ret
#     sleep 1
# done

# Change Number of Layers 
echo "Change Number of layers.."
for nl in {10..150..10}
do
    ret=$($cmd --nlayer $nl --ndim 2048 --batch_size 1024 2>/dev/null)
    echo $nl, $ret
    sleep 1
done
