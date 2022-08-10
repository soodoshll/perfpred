for bs in $(seq 10 10 300)
do
    cmd="python run_vgg.py --amp 1 "$bs
    mem=$(eval LD_PRELOAD=./fake_libcudart.so PYTORCH_NO_CUDA_MEMORY_CACHING=1 $cmd "2>/dev/null")
    ret=$?
    mem=$(tail -n 1 <<< $mem)
    if (($ret == 0))
    then
        mem1=$mem
    else
        mem1="fail"
    fi
    
    mem=$(eval $cmd "2>/dev/null")
    ret=$?
    mem=$(tail -n 1 <<< $mem)
    if (($ret == 0))
    then
        mem2=$mem
    else
        mem2="fail"
    fi

    echo $bs, $mem1, $mem2
done

# for n_layers in $(seq 50 10 100)
# do
#     cmd="python fake_alloc_mlp.py "$n_layers
#     mem=$(eval LD_PRELOAD=./fake_libcudart.so PYTORCH_NO_CUDA_MEMORY_CACHING=1 $cmd "2>/dev/null")
#     ret=$?
#     mem=$(tail -n 1 <<< $mem)
#     if (($ret == 0))
#     then
#         mem1=$mem
#     else
#         mem1="fail"
#     fi
    
#     mem=$(eval $cmd "2>/dev/null")
#     ret=$?
#     mem=$(tail -n 1 <<< $mem)
#     if (($ret == 0))
#     then
#         mem2=$mem
#     else
#         mem2="fail"
#     fi

#     echo $n_layers, $mem1, $mem2
# done