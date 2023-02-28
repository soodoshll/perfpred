# Performance Prediction of Deep Learning Models with Hardware-ware Optimizations

**Environment**

- OS: Ubuntu 20.04
- CUDA 11.6
- PyTorch 1.12
- Tested Hardware: RTX 2080Ti, RTX 3090, T4, V100

## Time Prediction

### NN-based Operator Runtime Predictor

We train a NN-based runtime predictor for complicated (non-elementwise) operators like ``conv2d``, ``linear``, `batchnorm`, `maxpool` and `bmm`.

**Performance Data Collection**

## Memory Prediction



## Misc
python dnnperf/fake_runner.py 1003 115 224
set cuda api_failures stop