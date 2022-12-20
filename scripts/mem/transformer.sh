#!bin/bash
cmd="python scripts/mem/transformer.py "$*
echo $cmd
if [ "$FAKE_ALLOC" == 1 ]
then
    echo "Using fake allocator"
    source ./faketorch/bin/activate
    cmd="env LD_PRELOAD=./fake_libcudart.so "$cmd
fi

for bs in {1..32..1} ; do
    model=bert-base-uncased
    ret=$($cmd --model $model --seq_len 512 --batch_size $bs 2>/dev/null)
    echo $model, $bs, $ret
    sleep 1
done

# for bs in 2 4 8
# do
#     model=bert-large-uncased
#     ret=$($cmd --model $model --seq_len 512 --batch_size $bs 2>/dev/null)
#     echo $model, $bs, $ret
#     sleep 1
# done

# for bs in 2 4 6
# do
#     model=gpt2-medium
#     ret=$($cmd --model $model --seq_len 512 --batch_size $bs 2>/dev/null)
#     echo $model, $bs, $ret
#     sleep 1
# done

# for bs in 4 8 12
# do
#     model=albert-large-v2
#     ret=$($cmd --model $model --seq_len 512 --batch_size $bs 2>/dev/null)
#     echo $model, $bs, $ret
#     sleep 1
# done