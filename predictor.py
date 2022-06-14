class Value(object):
    pass

class Variable(Value):
    pass

class Constant(Value):
    pass

class PlaceHolder(Value):
    pass

class GPUKernelEvent():
    def launch_time():
        pass

    def duration():
        pass

class CPUPredictor():
    def __init__(self):
        self.margin_before = 0
        self.margin_after = 0
        self.gpu_kernels = []
    
    def cpu_time(self):
        return