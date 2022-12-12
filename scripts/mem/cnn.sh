#!bin/bash
cmd="python scripts/mem/cnn.py"
if [ "$FAKE_ALLOC" == 1 ]
then
    echo "Using fake allocator"
    source ./faketorch/bin/activate
    cmd="env LD_PRELOAD=./fake_libcudart.so "$cmd
fi

# echo "Resnet"
# for bs in {16..256..16}
# do
#     ret=$($cmd resnet50 --batch_size $bs 2>/dev/null)
#     echo $bs, $ret
#     sleep 1
# done

echo "Densenet"
for bs in {4..96..4}
do
    ret=$($cmd densenet201 --batch_size $bs 2>/dev/null)
    echo $bs, $ret
    sleep 1
done

# echo "Change Number of layers.."
# for nl in {10..150..10}
# do
#     ret=$($cmd --nlayer $nl --ndim 4096 --batch_size 1024 2>/dev/null)
#     echo $nl, $ret
#     sleep 1
# done
