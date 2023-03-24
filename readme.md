# Performance Prediction of Deep Learning Models with Hardware-ware Optimizations

**Environment**

- OS: Ubuntu 20.04
- CUDA 11.6
- PyTorch 1.12
- Tested Hardware: RTX 2080Ti, RTX 3090, T4, V100

**Initializing Environment Variables**

```
source init.sh
```

## Time Prediction

### NN-based Operator Runtime Predictor

We train a NN-based runtime predictor for complicated (non-elementwise) operators like ``conv2d``, ``linear``, `batchnorm`, `maxpool` and `bmm`.

**Performance Data Collection**

```
python perfpred/measure.py [--num_gpus 1] [--use_amp] [--data-dir ./data] [--cooldown 0.01]
```

**Model Training**

```
python perfpred/predictor.py {conv2d,mm,batchnorm,maxpool2d,bmm}
```

### Tracing and Prediction

## Memory Prediction

We need a modified version of PyTorch, which makes two changes to PyTorch:

1. Replace the memory allocator with a dummy allocator;
1. Stop launching most kernels.

After all, we still have an option of non-intrusive mode of memory predictor. However, it might not be as accurate as the intrusive one.

**Building fake cuda runtime**
```
bash scripts/build.sh
```

## DNNPerf

As an important baseline we compared to, we reimplemented DNNPerf.

## Misc
python dnnperf/fake_runner.py 1003 115 224
set cuda api_failures stop