#!bin/bash
cmd="python scripts/mem/cnn.py "$*
echo $cmd
if [ "$FAKE_ALLOC" == 1 ]
then
    echo "Using fake allocator"
    source ./faketorch/bin/activate
    cmd="env LD_PRELOAD=./fake_libcudart.so "$cmd
fi

for bs in 32 64 128
do
    ret=$($cmd resnet50 --batch_size $bs 2>/dev/null)
    echo resnet50, $bs, $ret
    sleep 1
done

for bs in 32 64 96
do
    ret=$($cmd densenet201 --batch_size $bs 2>/dev/null)
    echo densenet201, $bs, $ret
    sleep 1
done

for bs in 16 32 64
do
    ret=$($cmd inception_v3 --batch_size $bs 2>/dev/null)
    echo inception_v3, $bs, $ret
    sleep 1
done

# echo "Densenet"
# for bs in {4..96..4}
# do
#     ret=$($cmd densenet201 --batch_size $bs 2>/dev/null)
#     echo $bs, $ret
#     sleep 1
# done

# echo "Change Number of layers.."
# for nl in {10..150..10}
# do
#     ret=$($cmd --nlayer $nl --ndim 4096 --batch_size 1024 2>/dev/null)
#     echo $nl, $ret
#     sleep 1
# done
