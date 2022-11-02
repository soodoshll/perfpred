from trace import Graph
import torch
import torchvision

device = torch.device('cuda')

"""
We support
* AlexNet
* DenseNet
* INCEPTION
* mobilenet_v2
* resnet
* squeezenet
* vgg
"""

model_list = [
    'alexnet',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'inception_v3',
    'mobilenet_v2',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'squeezenet1_0',
    'squeezenet1_1',
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn'
]

batch_size_list = [16, 32, 48, 64, 80, 96, 112, 128]
input_channels_list = [1, 3, 5, 7, 9]

print(len(model_list) * len(batch_size_list) * len(input_channels_list))

model = torchvision.models.vgg11()
model.to(device)

x = torch.rand((2, 3, 299, 299), device=device)
out = model(x)
g = Graph(out)
g.encode()