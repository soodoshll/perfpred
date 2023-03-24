import torch

def get_cv_example(model_name, batch_size):
    is_inception = model_name == "inception_v3"
    image_size = 299 if is_inception else 224
    import torchvision
    from .utils import torch_vision_model_revise, change_inplace_to_false
    torch_vision_model_revise(torchvision)
    model = getattr(torchvision.models, model_name)().cuda()
    model.apply(change_inplace_to_false)
    model.train()
    inputs = torch.empty((batch_size, 3, image_size, image_size)).cuda()
    labels = torch.zeros((batch_size, ), dtype=torch.int64).cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    def fn():
        torch.cuda.synchronize()
        optim.zero_grad()
        out = model(inputs)
        if is_inception:
            out = out[0]
        loss = loss_fn(out, labels)
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
    return fn
