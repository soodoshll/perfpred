import torch
def timing(func, nitr=1):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nitr):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / nitr