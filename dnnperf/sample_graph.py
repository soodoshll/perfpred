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
model = torchvision.models.squeezenet1_0()
model.to(device)

x = torch.rand((2, 3, 299, 299), device=device)
out = model(x)
# print(dir(out))
Graph(out)