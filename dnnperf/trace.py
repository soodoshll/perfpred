import torch
import numpy as np
import ctypes

def _tensor_size(a):
    return a.element_size() * a.nelement()

class Node(object):
    def __init__(self):
        self.out_edges = []
        self.in_edges = []
        self.weights = set()

    def add_out_edge(self, u, tensor_size=0, is_forward=True):
        self.out_edges.append([u, tensor_size, is_forward])
        u.in_edges.append(self)
    
    def fill_edge_feature(self):
        for edge in self.out_edges:
            edge[1] = self.output_size
    
    def backward(self):
        raise RuntimeError("何もない！" + str(self))

    def hyperparameter_info(self):
        raise RuntimeError("大変" + str(self))

    def tensor_info(self):
        raise RuntimeError("大変" + str(self))

class BPNode(Node):
    def __init__(self, fw):
        super(BPNode, self).__init__()
        self.forward = fw

    def hyperparameter_info(self):
        return self.forward.hyperparameter_info()

    def tensor_info(self):
        return self.forward.tensor_info()

class ConvNode(Node):
    def __init__(self, fn):
        super(ConvNode, self).__init__()
        self.input = fn._saved_input
        self.weight = fn._saved_weight
        # self.weights = [self.weight]
        self.padding = fn._saved_padding
        self.stride = fn._saved_stride
        self.dil = fn._saved_dilation

        self.input_shape = self.input.shape
        self.input_size = _tensor_size(self.input)
        self.weight_size = _tensor_size(self.weight)

        batch_size = self.input.shape[0]
        in_channels = self.input.shape[1]
        in_w = self.input.shape[2]
        in_h = self.input.shape[3]

        out_channels = self.weight.shape[0]
        k_w, k_h = self.weight.shape[2:]
        p_w, p_h = self.padding
        s_w, s_h = self.stride
        d_w, d_h = self.dil

        element_size = self.input.element_size()
        output_w = int((in_w + 2 * p_w - d_w * (k_w - 1)) -1 / s_w + 1)
        output_h = int((in_h + 2 * p_h - d_h * (k_h - 1)) -1 / s_h + 1)
        
        self.workspace_size = element_size * (in_channels * in_w * in_h + 1) * out_channels
        self.output_shape = (batch_size, out_channels, output_w, output_h)
        self.output_size = element_size * batch_size * out_channels * output_w * output_h

        self.hyperparameters = [
            batch_size, in_channels, out_channels, in_w, in_h, k_w, k_h, s_w, s_h, p_w, p_h, d_w, d_h,
        ]

        # print(self.hyperparameter_info())

    def backward(self):
        return ConvBPNode(self)

    def hyperparameter_info(self):
        return torch.tensor(self.hyperparameters)

    def tensor_info(self):
        return torch.tensor([self.input_size, self.output_size, _tensor_size(self.weight), self.workspace_size])

class ConvBPNode(BPNode):
    pass

class AddNode(Node):
    def __init__(self, fn):
        super(AddNode, self).__init__()
        # print(dir(fn))
        # we don't know the input shape here
    
    def fill_edge_feature(self):
        inputs = [src.output_size for src in self.in_edges]
        self.input_size = sum(inputs)
        self.output_size = max(inputs)
    
    def backward(self):
        return AddBPNode(self)

    def hyperparameter_info(self):
        return torch.tensor([])
    
    def tensor_info(self):
        inputs = [src.output_size for src in self.in_edges] 
        for w in self.weights:
            inputs.append(_tensor_size(w))
        return torch.tensor(inputs + [self.output_size])        

class AddBPNode(BPNode):
    pass

class ActivationNode(Node):
    def __init__(self, fn):
        super(ActivationNode, self).__init__()

        # print(dir(fn))
        if hasattr(fn, '_saved_result'):
            self.result = fn._saved_result
        elif hasattr(fn, '_saved_self'):
            self.result = fn._saved_self
        else:
            raise RuntimeError()

        self.input_shape = self.output_shape = self.result.shape
        self.input_size = self.output_size = _tensor_size(self.result)
    
    def backward(self):
        return ActivationBPNode(self)

    def hyperparameter_info(self):
        return torch.tensor([])
    
    def tensor_info(self):
        return torch.tensor([self.input_size, self.output_size])

class ActivationBPNode(BPNode):
    pass

class BatchNormNode(Node):
    def __init__(self, fn):
        super(BatchNormNode, self).__init__()
        self.input = fn._saved_input
        self.result1 = fn._saved_result1
        self.result2 = fn._saved_result2
        self.result3 = fn._saved_result3
        self.running_mean = fn._saved_running_mean
        self.running_var = fn._saved_running_var
        self.training = fn._saved_training
        self.weight = fn._saved_weight
        
        self.input_shape = self.output_shape = self.input.shape
        self.input_size = self.output_size = _tensor_size(self.input)

    def backward(self):
        return BatchNormBPNode(self)
    
    def hyperparameter_info(self):
        return torch.tensor([])
    
    def tensor_info(self):
        return torch.tensor([self.input_size, self.output_size])

class BatchNormBPNode(BPNode):
    pass

class MaxpoolNode(Node):
    def __init__(self, fn):
        super(MaxpoolNode, self).__init__()
        self.input = fn._saved_self
        self.input_size = _tensor_size(self.input)

        batch_size = self.input.shape[0]
        in_channels = self.input.shape[1]
        in_w = self.input.shape[2]
        in_h = self.input.shape[3]

        d_w, d_h = fn._saved_dilation
        k_w, k_h = fn._saved_kernel_size
        p_w, p_h = fn._saved_padding
        s_w, s_h = fn._saved_stride
        element_size = self.input.element_size()
        output_w = int((in_w + 2 * p_w - d_w * (k_w - 1)) -1 / s_w + 1)
        output_h = int((in_h + 2 * p_h - d_h * (k_h - 1)) -1 / s_h + 1)
        self.output_size = element_size * batch_size * in_channels * output_w * output_h 

        self.hyperparameter = [batch_size, in_channels, in_w, in_h, d_w, d_h, k_w, k_h, p_w, p_h, s_w, s_h]
    
    def backward(self):
        return MaxpoolBPNode(self)
    
    def hyperparameter_info(self):
        return torch.tensor(self.hyperparameter)
    
    def tensor_info(self):
        return torch.tensor([self.input_size, self.output_size])

class MaxpoolBPNode(BPNode):
    pass

class AddmmNode(Node):
    def __init__(self, fn): 
        super(AddmmNode, self).__init__()
        self.mat1 = fn._saved_mat1
        self.mat2 = fn._saved_mat2
        self.input_size = _tensor_size(self.mat1) + _tensor_size(self.mat2)
        
        element_size = self.mat1.element_size()
        self.output_size = element_size * self.mat1.shape[0] * self.mat2.shape[1]
        self.hyperparameter = [self.mat1.shape[0], self.mat1.shape[1], self.mat2.shape[1]]

    def backward(self):
        return AddmmBPNode(self)

    def hyperparameter_info(self):
        return torch.tensor(self.hyperparameter)
    
    def tensor_info(self):
        return torch.tensor([_tensor_size(self.mat1), _tensor_size(self.mat2), self.output_size])

class AddmmBPNode(BPNode):
    pass

class AccumulateGradNode(Node):
    def __init__(self, fn): 
        super(AccumulateGradNode, self).__init__()
        self.var = fn.variable
        self.input_size = self.output_size = _tensor_size(fn.variable)

class ReshapeNode(Node):
    def  __init__(self, fn):
        super(ReshapeNode, self).__init__()
    
    def fill_edge_feature(self):
        assert(len(self.in_edges) == 1)
        self.input_size = self.output_size = self.in_edges[0].output_size

class MeanNode(Node):
    def __init__(self, fn):
        super(MeanNode, self).__init__()
        self.dim =[ctypes.c_long(d).value for d in fn._saved_dim]
        self.input_size = self.output_size = 4 * np.prod(fn._saved_self_sizes)
        for d in self.dim:
            self.output_size //= fn._saved_self_sizes[d]
    
    def backward(self):
        return MeanBPNode(self)

    def hyperparameter_info(self):
        return torch.tensor([])
    
    def tensor_info(self):
        return torch.tensor([self.input_size, self.output_size])

class MeanBPNode(BPNode):
    pass

class TNode(Node):
    def __init__(self, fn):
        super(TNode, self).__init__()
    
    def fill_edge_feature(self):
        assert(len(self.in_edges) == 1)
        self.input_size = self.output_size = self.in_edges[0].output_size

class SGDNode(Node):
    def __init__(self):
        super(SGDNode, self).__init__() 
        self.weight_size = 0
    
    def fill_edge_feature(self):
        raise RuntimeError("駄目！")
    
    def hyperparameter_info(self):
        return torch.tensor([])
    
    def add_weight_size(self, ws):
        self.weight_size += ws

    def tensor_info(self):
        return torch.tensor(self.weight_size)

class DropoutNode(Node):
    def __init__(self, fn):
        super(DropoutNode, self).__init__()
        self.output_shape = fn._saved_result1.shape
        self.input_size = self.output_size = _tensor_size(fn._saved_result1)
    
    def backward(self):
        return DropoutBPNode(self)
    
    def hyperparameter_info(self):
        return torch.tensor([])
    
    def tensor_info(self):
        return torch.tensor([self.inputs_size, self.output_size])

class DropoutBPNode(BPNode):
    pass

class AdaptiveAvgPoolNode(Node):
    def __init__(self, fn):
        super(AdaptiveAvgPoolNode, self).__init__()
        self.input_size = self.output_size = _tensor_size(fn._saved_self)

    def backward(self):
        return AdapativeAvgPoolBPNode(self)

    def hyperparameter_info(self):
        return torch.tensor([])
    
    def tensor_info(self):
        return torch.tensor([self.inputs_size, self.output_size])

class AdapativeAvgPoolBPNode(BPNode):
    pass

class CatNode(Node):
    def __init__(self, fn):
        super(CatNode, self).__init__()
        # print(dir(fn))
        # we don't know the input shape here
    
    def fill_edge_feature(self):
        inputs = [src.output_size for src in self.in_edges]
        self.input_size = sum(inputs)
        self.output_size = sum(inputs)
    
    def backward(self):
        return CatBPNode(self)

    def hyperparameter_info(self):
        return torch.tensor([])
    
    def tensor_info(self):
        inputs = [src.output_size for src in self.in_edges] 
        for w in self.weights:
            inputs.append(_tensor_size(w))
        return torch.tensor(inputs + [self.output_size])        

class CatBPNode(BPNode):
    pass

class AvgPoolNode(Node):
    def __init__(self, fn):
        super(AvgPoolNode, self).__init__()
        self.input_size = _tensor_size(fn._saved_self)
        self.input_shape = fn._saved_self.shape
        k_w, k_h = fn._saved_kernel_size
        s_w, s_h = fn._saved_stride
        p_w, p_h = fn._saved_padding
        
        batch_size = self.input_shape[0]
        channels = self.input_shape[1]
        in_w = self.input_shape[2]
        in_h = self.input_shape[3]

        element_size = fn._saved_self.element_size()
        out_w = (in_w + 2*p_w - (k_w - 1)) // s_w
        out_h = (in_h + 2*p_h - (k_h - 1)) // s_h
        self.output_size = element_size * batch_size * out_w * out_h * channels

        self.hyperparameters = [batch_size, channels, in_w, in_h, k_w, k_h, s_w, s_h, p_w, p_h]


    def backward(self):
        return AvgPoolBPNode(self)

    def hyperparameter_info(self):
        return self.hyperparameters
    
    def tensor_info(self):
        return torch.tensor([self.input_size, self.output_size])

class AvgPoolBPNode(BPNode):
    pass

class Graph(object):
    name_to_constructor = {
        'ConvolutionBackward' : ConvNode,
        'AddBackward' : AddNode,
        'AddmmBackward' : AddmmNode,
        'ReluBackward' : ActivationNode,
        'BatchNorm' : BatchNormNode,
        'MaxPool' : MaxpoolNode,
        'AccumulateGrad' : AccumulateGradNode,
        'ReshapeAlias' : ReshapeNode,
        'Mean' : MeanNode,
        'TBackward' : TNode,

        'NativeDropoutBackward' : DropoutNode,
        'AdaptiveAvgPool' : AdaptiveAvgPoolNode,
        'CatBackward' : CatNode,
        'AvgPool2D': AvgPoolNode,
        'HardtanhBackward0' : ActivationNode,
        'ViewBackward' : TNode,
        # 'CloneBackward' : TNode,
        'TransposeBackward0' : TNode,
        'AsStridedBackward' : TNode,

        # 'MulBackward0' : AddmmNode
        # Split
        }
    
    node_types = [
        ConvNode, ConvBPNode, AddNode, AddBPNode, AddmmNode, AddmmBPNode,
        ActivationNode, ActivationBPNode, BatchNormNode, BatchNormBPNode,
        MaxpoolNode, MaxpoolBPNode, ReshapeNode, MeanNode, MeanBPNode,
        SGDNode,
        DropoutNode, DropoutBPNode,
        AdaptiveAvgPoolNode, AdapativeAvgPoolBPNode,
        CatNode, CatBPNode,
        AvgPoolNode, AvgPoolBPNode
    ]

    max_hyperparam_feature_length = 15
    max_tensor_feature_length = 5

    def __init__(self, out):
        self.out = out.grad_fn
        self.nodes = []
        self.fn_to_node = {}

        self.traverse(out.grad_fn)
        self.remove_t_and_reshape()
        self.remove_accumulate_grad()

        # for node in self.nodes:
        #     for d in node.in_edges:
        #         print(d, '--->', node)

        self.fill_edge_feature()

        self.generate_backward_graph()
    
    def remove_t_and_reshape(self):
        topo_sort_ret = self.topo_sort()
        assert len(topo_sort_ret) == len(self.nodes)
        new_nodes = []
        for node in topo_sort_ret:
            if isinstance(node, TNode) or isinstance(node, ReshapeNode):
                # print(node)
                assert len(node.in_edges) == 1
                src = node.in_edges[0]
                src.out_edges = node.out_edges
                # if self.out == node:
                #     # print("BBQle")
                #     self.out = src
                for dst, _, _ in node.out_edges:
                    # print(dst)
                    for i, pred in enumerate(dst.in_edges):
                        if pred == node:
                            dst.in_edges[i] = src
            else:
                new_nodes.append(node)
        self.nodes = new_nodes
    
    def remove_accumulate_grad(self):
        topo_sort_ret = self.topo_sort()
        new_nodes = []
        for node in topo_sort_ret:
            if isinstance(node, AccumulateGradNode):
                assert len(node.in_edges) == 0
                for dst, _, _ in node.out_edges:
                    if not (node.var.data in dst.weights):
                        dst.weights.add(node.var.data)
                    for i, pred in enumerate(dst.in_edges):
                        if pred == node:
                            del dst.in_edges[i]
                            break
            else:
                new_nodes.append(node)
        self.nodes = new_nodes

    def traverse(self, fn):
        if fn is None:
            return
        if fn in self.fn_to_node:
            return self.fn_to_node[fn]
        else:
            key = None
            for k in self.name_to_constructor.keys():
                if str(fn).find(k) >= 0:
                    key = k
            if key is None:
                raise RuntimeError("unsupported op " + str(fn))
            node = self.name_to_constructor[key](fn)
            for next_fn in fn.next_functions:
                next_node = self.traverse(next_fn[0])
                if next_node is not None:
                    next_node.add_out_edge(node)
            self.fn_to_node[fn] = node
            self.add_node(node)
            return node

    def add_node(self, node):
        self.nodes.append(node)
    
    def topo_sort(self):
        in_degree = {}
        q = []
        ptr = 0
        for node in self.nodes:
            in_degree[node] = len(node.in_edges)
            if in_degree[node] == 0:
                q.append(node)
        while ptr < len(self.nodes):
            cur = q[ptr]
            for child, _, _ in cur.out_edges:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    q.append(child)
            ptr += 1
        return q

    def fill_edge_feature(self):
        topo_sort_ret = self.topo_sort()
        for node in topo_sort_ret:
            node.fill_edge_feature()
    
    def generate_backward_graph(self):
        self.optim = SGDNode()
        self.add_node(self.optim)
        fw_bw_map = {}
        def backward(node):
            # print(node)
            if node in fw_bw_map:
                return fw_bw_map[node]
            
            bw_node = node.backward()
            self.add_node(bw_node)
            fw_bw_map[node] = bw_node
            for pred in node.in_edges:
                pred.add_out_edge(bw_node, pred.output_size, False)
                pred_bw = backward(pred)
                bw_node.add_out_edge(pred_bw, pred.output_size, False)
            weight_size = sum([_tensor_size(w) for w in node.weights])
            bw_node.add_out_edge(self.optim, weight_size, False)
            self.optim.add_weight_size(weight_size)
            return bw_node

        out = self.fn_to_node[self.out]
        if isinstance(out, ReshapeNode):
            out = out.in_edges[0]
        backward(out)

    def encode(self):
        node_to_id = {}
        for idx, node in enumerate(self.nodes):
            node_to_id[node] = idx
        edge_list = []
        edge_data = []
        for node in self.nodes:
            src_id = node_to_id[node]
            for dst, data_size, is_forward in node.out_edges:
                edge_list.append([src_id, node_to_id[dst]])
                edge_data.append([is_forward, data_size])
        
        node_data = []
        node_type = []
        for node in self.nodes:
            type_id = self.node_types.index(type(node))
            node_type.append(type_id)
            # remember to do one-hot when loading data

            h_feat = node.hyperparameter_info()
            t_feat = node.tensor_info()

            h_feat_align = torch.zeros((self.max_hyperparam_feature_length,))
            if len(h_feat.shape) > 0:
                h_feat_align[:len(h_feat)] = h_feat

            t_feat_align = torch.zeros((self.max_tensor_feature_length, ))
            if len(t_feat.shape) > 0:
                t_feat_align[:len(t_feat)] = t_feat

            node_data.append(torch.concat([h_feat_align, t_feat_align]))
        
        return node_type, node_data, edge_list, edge_data

    def dump(self, filename):
        encoded_ret  = self.encode()
        torch.save(encoded_ret, filename)