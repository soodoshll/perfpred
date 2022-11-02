import sqlite3
# from timeit import default_timer
# from unittest import defaultTestLoader
import torch
from torch import nn
from tqdm import trange, tqdm
import random
import time
import numpy as np
from functools import partial
# import xgboost as xgb
# from torchmetrics import MeanAbsolutePercentageError
import glob, pickle
import argparse
import sys
# from matplotlib import pyplot as plt

LINEAR_PATH = glob.glob("./data/matmul_*.data")
CONV2D_PATH_SQL = ["./habitat-data/conv2d/conv2d-RTX2080Ti-0.sqlite", "./habitat-data/conv2d/conv2d-RTX2080Ti-1.sqlite"]
CONV2D_PATH = glob.glob("./data/eco*/conv_*.data") + glob.glob("./data/conv_2080ti*.data")
MAXPOOL_PATH = glob.glob("./data/maxpool_*.data")
BATCHNORM_PATH = glob.glob("./data/batchnorm_*.data")
BMM_PATH = glob.glob("./data/bmm_*.data")

device = torch.device('cuda')

class PeriodicActivation(nn.Module):
    def forward(self, x):
        return x + (torch.sin(x)) 

class AbsLayer(nn.Module):
    def forward(self, x):
        return torch.exp(x)

class ExpLayer(nn.Module):
    def forward(self, x):
        return torch.exp(x) 

def make_mlp(device, input_dim, hidden_layers=[1024] * 8  , activation=nn.ReLU):
    layers = []
    last = input_dim
    for idx, h in enumerate(hidden_layers):
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    layers.append(nn.Linear(last, 1))
    # layers.append(ExpLayer())
    model = nn.Sequential(*layers)
    model.to(device)    
    return model

def NormL1(output, target, inputs):
    bs = inputs[:, 1]
    image_size = inputs[:, 3]
    in_channels = inputs[:, 4]
    out_channels = inputs[:, 5]
    kernel_size = inputs[:, 6]
    stride = inputs[:, 7]
    is_forward = inputs[:, 9]

    c = 1 / bs * (stride **2) / (kernel_size ** 2)
    target = target * c
    output = output * c
    return torch.mean(torch.abs(target - output))

def MAPELoss(output, target, inputs=None):
    loss = torch.abs((target - output) / target)
    return torch.mean(loss)    

def MSPELoss(output, target, inputs=None):
    loss = (target - output) / target
    loss = loss * loss
    return torch.mean(loss)  

def LogMSELoss(output, target, inputs=None):
    loss = torch.log(target) - output
    loss = (loss * loss)
    return torch.mean(loss)

def ExpLoss(output, target, inputs=None):
    loss = torch.abs(torch.exp(output/target) - torch.e)
    return torch.mean(loss)

class Predictor(object):
    def load_data_sql(self, paths):
        rows = []
        for path in paths:
            conn = sqlite3.connect(path)
            c = conn.cursor()
            cursor = c.execute("SELECT * from recordings")
            rows.extend([row[1:] for row in cursor])
        self.raw_dataset = torch.tensor(rows)
        print("datasize:", len(rows))
        
        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.avgs[-1] = 0
        self.stds[-1] = 1
        self.dataset = (self.raw_dataset - self.avgs) / self.stds
        print("avg:", self.avgs)
        print("std:", self.stds)  

        train_set_size = int(self.dataset.shape[0] * 0.8)
        test_set_size = self.dataset.shape[0] - train_set_size
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_set_size, test_set_size])

    def train(self, model_path, batch_size=512, num_epoch=30, hooks=[]):
        model = self.model
        model.to(self.device)
        model.train()
        dataloader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=8)
        
        loss_fn = MSPELoss
        optim = torch.optim.Adam(
                model.parameters(), 
                lr=1e-4, 
                weight_decay=1e-5
                )
        lowest_err = 9e9
        for epoch_idx in range(num_epoch):
            for data in dataloader:
                inputs = self.preprocess(data[:, :-1]).to(self.device)
                # print(f"mem: {torch.cuda.memory_allocated() /1e6} MB")
                labels = data[:, -1].to(self.device)
                optim.zero_grad(set_to_none=True)
                out = model(inputs)[:, 0]
                loss = loss_fn(out, labels, inputs)
                loss.backward()
                optim.step()

            if epoch_idx % 10 == 0:
                for hook in hooks:
                    hook()
                test_error = self.test_set_error()
                print(f"[{epoch_idx}/{num_epoch}] \t train: {loss.detach().cpu().numpy() : .5f} test: {test_error : .5f} | truth: {labels[0] :.2f}, pred: {out[0] :.2f}", file=sys.stderr)
                # early stop
                if epoch_idx == 0 or test_error < lowest_err:
                    lowest_err = test_error
                    torch.save(
                        {"model_state_dict" : model.state_dict(),
                        "avg" : self.avgs,
                        "std" : self.stds,
                        }, model_path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.avgs = checkpoint['avg']
        self.stds = checkpoint['std']

    def predict(self, inputs):
        self.model.to(torch.device('cpu'))
        self.model.eval()
        inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = self.preprocess(inputs)
        out = self.model(inputs)
        out = out[0] * self.stds[-1] + self.avgs[-1]
        # out = torch.exp(out)
        return out.detach().numpy()

    def test_set_error(self, batch_size=1000, filename=None):
        # print(len(self.test_set))
        dataloader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=True) 
        errors = []
        # print(f"    mem: {torch.cuda.memory_allocated() /1e6} MB")
        with torch.no_grad():
            for data in dataloader:
                # print(f"    mem: {torch.cuda.memory_allocated() /1e6} MB")
                inputs = self.preprocess(data[:, :-1]).to(self.device)
                labels = data[:, -1].to(self.device)
                out = self.model(inputs)
                pred = out[:, 0] * self.stds[-1] + self.avgs[-1]
                # pred = torch.exp(pred)
                truth = labels * self.stds[-1] + self.avgs[-1]
                # print(pred[0], truth[0])
                error = (pred - truth) / truth
                errors.append(error)
        errors = torch.concat(errors)
        errors = errors.cpu().detach().numpy()
        # if filename is not None:
        #     plt.hist(errors, bins=100)
        #     plt.savefig(filename)

        return np.mean(np.abs(errors))

    def preprocess(self, data):
        return (data - self.avgs[:-1]) / self.stds[:-1]

class LinearPredictor(Predictor):
    def __init__(self, device=torch.device('cpu')):
        self.feature_name = ['bias', 'batch', 'in_features', 'out_features', 'is_forward', 'use_fp16']
        self.model = make_mlp(device, len(self.feature_name))
        self.device = device

    def load_data(self, filenames):
        rows_fp16 = []
        rows_fp32 = []
        for fn in filenames:
            with open(fn, "rb") as f:
                while 1:
                    try:
                        objs = pickle.load(f)
                        for obj in objs:
                            dur_forward, dur_backward, use_fp16, batch_size, in_features, out_features, bias = obj
                            rows = rows_fp16 if use_fp16 else rows_fp32
                            rows.append(
                                (bias, batch_size, in_features, out_features, 1, use_fp16, dur_forward)
                            )
                            rows.append(
                                (bias, batch_size, in_features, out_features, 0, use_fp16, dur_backward)
                            )
                    except (EOFError):
                        break
        min_len = min(len(rows_fp16), len(rows_fp32))
        print(f"fp16: {len(rows_fp16)} | fp32 : {len(rows_fp32)}")
        rows = rows_fp16[-min_len:] + rows_fp32[-min_len:]
        self.raw_dataset = torch.tensor(rows, dtype=torch.float32)
        print(self.raw_dataset[0])
        print("datasize:", len(rows))

        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.avgs[-1] = 0
        self.stds[-1] = 1

        self.avgs[-2] = 0
        self.stds[-2] = 1

        self.dataset = self.raw_dataset
        print("avg:", self.avgs)
        print("std:", self.stds)  

        train_set_size = int(self.dataset.shape[0] * 0.8)
        test_set_size = self.dataset.shape[0] - train_set_size
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_set_size, test_set_size]) 

    def test_auto(self, nitr=100):
        self.model.to(torch.device('cpu'))
        for _ in range(nitr):
            n = random.randint(16, 256)
            m = random.randint(1, 4096)
            k = random.randint(1, 4096)
            # n, m, k = 249, 10060, 157
            # n, m, k = 877, 7129, 8717

            A = torch.rand((n, m), device=device, dtype=torch.float32)
            layer = nn.Linear(m, k, device=device)

            for _ in range(3):
                layer(A)
            torch.cuda.synchronize()
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            t0 = time.time()
            for _ in range(10):
                layer(A)
            # end.record()
            torch.cuda.synchronize()
            # dur = (start.elapsed_time(end)) / 10
            dur1 = (time.time() - t0) / 10 * 1e3
            pred = self.predict([0, n, m, k, 1])
            print(dur1, pred)
            if dur1 > 10:
                print(n, m, k, dur1)
                return 

class BatchMatMulPredict(Predictor):
    def __init__(self, device=torch.device('cpu')):
        self.feature_name = ['batch', 'l', 'm', 'n', 'is_forward', 'use_fp16']
        self.model = make_mlp(device, len(self.feature_name))
        self.device = device

    def load_data(self, filenames):
        rows_fp16 = []
        rows_fp32 = []
        for fn in filenames:
            with open(fn, "rb") as f:
                while 1:
                    try:
                        objs = pickle.load(f)
                        for obj in objs:
                            dur_forward, dur_backward, use_fp16, batch_size, l, m, n = obj
                            rows = rows_fp16 if use_fp16 else rows_fp32
                            rows.append(
                                (batch_size, l, m, n, 1, use_fp16, dur_forward)
                            )
                            rows.append(
                                (batch_size, l, m, n, 0, use_fp16, dur_backward)
                            )
                    except (EOFError):
                        break
        min_len = min(len(rows_fp16), len(rows_fp32))
        print(f"fp16: {len(rows_fp16)} | fp32 : {len(rows_fp32)}")
        rows = rows_fp16[-min_len:] + rows_fp32[-min_len:]
        self.raw_dataset = torch.tensor(rows, dtype=torch.float32)
        print(self.raw_dataset[0])
        print("datasize:", len(rows))

        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.avgs[-1] = 0
        self.stds[-1] = 1

        self.avgs[-2] = 0
        self.stds[-2] = 1

        self.dataset = self.raw_dataset
        print("avg:", self.avgs)
        print("std:", self.stds)  

        train_set_size = int(self.dataset.shape[0] * 0.8)
        test_set_size = self.dataset.shape[0] - train_set_size
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_set_size, test_set_size]) 

def modpos(n, m, zero_base = True):
    n = n % m
    if zero_base:
        return n
    return m if n == 0 else n

class Conv2DPredictor(Predictor):
    def __init__(self, modulo=True, device=torch.device('cpu')):
        self.feature_name = ['bias', 'batch', 'image_size', 'in_channels', 'out_channels', 'kernel_size',
        'stride', 'padding', 'is_forward', 'use_fp16']

        self.device = device
        self.modulo = modulo
        if modulo:
            self.model = make_mlp(device, len(self.feature_name) + 32 + 4 + 64 + 64 + 7 + 7)
        else:
            self.model = make_mlp(device, len(self.feature_name))

        # self.xgb_r = xgb.XGBRegressor(objective ='reg:squarederror',
                #   n_estimators = 200, seed = 123)

    def shrink_training_set(self, n):
        # idx = torch.randperm(len(self.original_train_set))
        # idx = idx[:n].numpy()
        # print(idx.shape, type(idx[0]))
        # print(self.original_train_set[idx[0]])
        # print(type(self.original_train_set))
        # self.train_set = [self.original_train_set[i] for i in idx]
        self.train_set = self.original_train_set[:n]

    def load_data(self, filenames):
        # rows = []
        rows_fp16 = []
        rows_fp32 = []
        cnt = 0
        for fn in filenames:
            with open(fn, "rb") as f:
                while 1:
                    try:
                        objs = pickle.load(f)
                        for obj in objs:
                            if len(obj) == 10:
                                use_fp16 = False
                                dur_forward, dur_backward, dx, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = obj
                            else:
                                dur_forward, dur_backward, dx, use_fp16, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = obj
                            if 5 * dur_forward < dur_backward:
                                # print("eliminate outliners")
                                continue
                            # if kernel_size == 7 or kernel_size == 6:
                                # cnt += 1
                            # else:
                                # continue
                            rows = rows_fp16 if use_fp16 else rows_fp32
                            rows.append(
                                (0, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, use_fp16, dur_forward)
                            )
                            rows.append(
                                (0, batch_size, 
                                image_size, in_channels, out_channels, kernel_size, stride, padding, 0, use_fp16, dur_backward)
                            )
                    except (EOFError):
                        break
        print(cnt)
        min_len = min(len(rows_fp16), len(rows_fp32))
        # print(f"fp16: {len(rows_fp16)} | fp32: {len(rows_fp32)} | min len : {min_len}")
        rows = rows_fp16[-min_len:] + rows_fp32[-min_len:]

        # rows = rows_fp16

        # print(len(rows))
        self.raw_dataset = torch.tensor(rows, dtype=torch.float32)
        # print("max bs:", torch.max(self.raw_dataset[:, 1]))
        # print("datasize:", len(self.raw_dataset))

        self.dataset = self.raw_dataset 

        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.avgs[-1] = 0
        self.stds[-1] = 1

        ####
        self.avgs[0]= 0
        self.stds[0] = 1

        self.avgs[-2] = 0
        self.stds[-2] = 1

        # self.avgs[-3] = 0
        # self.stds[-3] = 1
        ####

        # print("avg:", self.avgs)
        # print("std:", self.stds)  

        train_set_size = int(self.dataset.shape[0] * 0.8)
        test_set_size = self.dataset.shape[0] - train_set_size

        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_set_size, test_set_size]) 
        # self.train_set = self.train_set[torch.randperm(len(self.train_set))]
        self.original_train_set = self.train_set

    def preprocess(self, data):
        if not self.modulo:
            return (data - self.avgs[:-1]) / self.stds[:-1]
        if len(data.shape) == 2:
            batchsize = data[:, 1].type(torch.int64)
            in_channels = data[:, 3].type(torch.int64)
            out_channels = data[:, 4].type(torch.int64)
            image_size = data[:, 2].type(torch.int64)
            kernel_size = data[:, 5].type(torch.int64)
            stride = data[:, 6].type(torch.int64)
        else:
            batchsize = data[1].type(torch.int64) 
            in_channels = data[3].type(torch.int64)
            out_channels = data[4].type(torch.int64)        
            image_size = data[2].type(torch.int64)
            kernel_size = data[5].type(torch.int64)
            stride = data[6].type(torch.int64)

        batchsize_mod = batchsize % 32
        batchsize_mod = nn.functional.one_hot(batchsize_mod, 32)

        in_channels_mod = (in_channels) % 64
        out_channels_mod = (out_channels) % 64
        image_size_mod = image_size % 4

        in_channels_mod = nn.functional.one_hot(in_channels_mod, 64)
        out_channels_mod = nn.functional.one_hot(out_channels_mod, 64)
        image_size_mod = nn.functional.one_hot(image_size_mod, 4)

        kernel_one_hot = nn.functional.one_hot(kernel_size - 1, 7)
        stride_one_hot = nn.functional.one_hot(stride - 1, 7)
        data = (data - self.avgs[:-1]) / self.stds[:-1]
        
        inputs = torch.concat((data, 
            batchsize_mod.type(torch.float32),
            in_channels_mod.type(torch.float32),
            out_channels_mod.type(torch.float32),
            image_size_mod.type(torch.float32),
            kernel_one_hot.type(torch.float32),
            stride_one_hot.type(torch.float32)
            ), axis=-1)
        # print(inputs.shape)
        return inputs

    def train_set_error(self, batch_size=1000):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True) 
        errors = []
        for data in dataloader:
            inputs = self.preprocess(data[:, :-1].to(device))
            labels = data[:, -1].to(device)
            out = self.model(inputs)
            pred = out[:, 0] * self.stds[-1] + self.avgs[-1]
            truth = labels * self.stds[-1] + self.avgs[-1]
            error = torch.abs(pred - truth) / truth
            errors.append(error)
        return torch.mean(torch.concat(errors))

    def xgb_fit(self):
        self.xgb_r.fit(self.dataset[:, :-1], self.raw_dataset[:, -1])

    def xgb_predict(self, input):
        input = torch.tensor(input, dtype=torch.float)
        input = (input - self.avgs[:-1]) / self.stds[:-1]
        pred = self.xgb_r.predict(np.array([input.numpy()]))
        return pred[0]

class MaxPoolingPredictor(Predictor):
    def __init__(self, device=torch.device('cpu')):
        self.feature_name = ['batch_size', 'kernel_size', 'image_size', 'channels', 'stride', 'is_forward', 'use_fp16']
        self.model = make_mlp(device, len(self.feature_name))
        self.device = device        

    def load_data(self, filenames):
        rows_fp16 = []
        rows_fp32 = []
        for fn in filenames:
            with open(fn, "rb") as f:
                while 1:
                    try:
                        objs = pickle.load(f)
                        for obj in objs:
                            # print(obj)
                            dur_forward, dur_backward, use_fp16, batch_size, kernel_size, image_size, channels, stride = obj
                            rows = rows_fp16 if use_fp16 else rows_fp32
                            rows.append(
                                (batch_size, kernel_size, image_size, channels, stride, 1, use_fp16, dur_forward)
                            )
                            rows.append(
                                (batch_size, kernel_size, image_size, channels, stride, 0, use_fp16, dur_backward)
                            )
                    except (EOFError):
                        break    
        min_len = min(len(rows_fp16), len(rows_fp32))
        print(f"fp16: {len(rows_fp16)} | fp32: {len(rows_fp32)} | min len : {min_len}")
        rows = rows_fp16[-min_len:] + rows_fp32[-min_len:] 
        self.raw_dataset = torch.tensor(rows, dtype=torch.float32)
        self.dataset = self.raw_dataset
        # print(self.raw_dataset[0])
        # print(self.raw_dataset[0])
        print("datasize:", len(rows))
        

        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.avgs[-1] = 0
        self.stds[-1] = 1

        self.avgs[-2] = 0
        self.stds[-2] = 1
        print("avg:", self.avgs)
        print("std:", self.stds)  

        train_set_size = int(self.dataset.shape[0] * 0.8)
        test_set_size = self.dataset.shape[0] - train_set_size
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_set_size, test_set_size]) 

class BatchNormPredictor(Predictor):
    def __init__(self, device=torch.device('cpu')):
        self.feature_name = ['batch_size', 'image_size', 'channels', 'is_forward', 'use_fp16']
        self.model = make_mlp(device, len(self.feature_name))
        self.device = device        
    
    def load_data(self, filenames):
        rows_fp16 = [] 
        rows_fp32 = []
        for fn in filenames:
            with open(fn, "rb") as f:
                while 1:
                    try:
                        objs = pickle.load(f)
                        for obj in objs:
                            # print(obj)
                            dur_forward, dur_backward, use_fp16, batch_size, image_size, channels = obj
                            rows = rows_fp16 if use_fp16 else rows_fp32
                            rows.append(
                                (batch_size, image_size, channels, 1, use_fp16, dur_forward)
                            )
                            rows.append(
                                (batch_size, image_size, channels, 0, use_fp16, dur_backward)
                            )
                    except (EOFError):
                        break    
        
        min_len = min(len(rows_fp16), len(rows_fp32))
        print(f"fp16: {len(rows_fp16)} | fp32 : {len(rows_fp32)}")
        rows = rows_fp16[-min_len:] + rows_fp32[-min_len:]        

        self.raw_dataset = torch.tensor(rows, dtype=torch.float32)
        self.dataset = self.raw_dataset
        print(self.raw_dataset[0])
        print(self.raw_dataset[0])
        print("datasize:", len(rows))

        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)

        self.avgs[-1] = 0
        self.stds[-1] = 1
 
        self.avgs[-2] = 0
        self.stds[-2] = 1
        print("avg:", self.avgs)
        print("std:", self.stds)  

        train_set_size = int(self.dataset.shape[0] * 0.8)
        test_set_size = self.dataset.shape[0] - train_set_size
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_set_size, test_set_size]) 

def train(args):
    # linear_pred = LinearPredictor()
    # linear_pred.load_data(LINEAR_PATH)
    # linear_pred.train('predictor_model_linear.th', 
    #                 batch_size=512,
    #                 num_epoch=80, 
    #                 hooks=[lambda : print("error on test set:", linear_pred.test_set_error())])
    # return
    modulo = not args.nomodulo
    if args.op == 'conv2d':
        conv_pred = Conv2DPredictor(modulo, device=device)
        conv_pred.load_data(filenames=CONV2D_PATH)
        print("train set size:", len(conv_pred.original_train_set))
        
        model_name = f"./model/predictor_model_conv2d{'_nomodulo' if not modulo else ''}.th"
        conv_pred.train(model_name,
                        batch_size=512,
                        num_epoch=300, 
                        hooks=[])
    elif args.op == "mm":
        linear_pred = LinearPredictor(device=device)
        linear_pred.load_data(filenames=LINEAR_PATH)
        print("train set size:", len(linear_pred.train_set))
        
        model_name = "./model/predictor_model_linear.th"
        linear_pred.train(model_name,
                          batch_size=512,
                          num_epoch=200)
    elif args.op == "batchnorm":
        batchnorm_pred = BatchNormPredictor(device=device)
        batchnorm_pred.load_data(filenames=BATCHNORM_PATH)
        print("train set size:", len(batchnorm_pred.train_set))
        
        model_name = "./model/predictor_model_batchnorm.th"
        batchnorm_pred.train(model_name,
                             batch_size=512,
                             num_epoch=200)
    elif args.op == "maxpool2d":
        maxpool_pred = MaxPoolingPredictor(device=device)
        maxpool_pred.load_data(filenames=MAXPOOL_PATH)
        print("train set size:", len(maxpool_pred.train_set))
        
        model_name = "./model/predictor_model_maxpool.th"
        maxpool_pred.train(model_name,
                           batch_size=512,
                           num_epoch=200)
    elif args.op == "bmm":
        bmm_pred = BatchMatMulPredict(device=device)
        bmm_pred.load_data(filenames=BMM_PATH)
        print("train set size:", len(bmm_pred.train_set))
        model_name = "./model/predictor_model_bmm.th"
        bmm_pred.train(model_name,
                       batch_size=512,
                       num_epoch=200)
    else:
        raise RuntimeError("Not supported")
    # error = conv_pred.test_set_error(filename="conv_error.png")

    # error = conv_pred.train_set_error()
    # maxpool_pred = MaxPoolingPredictor()
    # maxpool_pred.load_data(MAXPOOL_PATH)
    # maxpool_pred.train("predictor_model_maxpool.th",
    #                     batch_size=512,
    #                     num_epoch=200,
    #                     hooks=[lambda : print("error on test set:", maxpool_pred.test_set_error())])

    # batchnorm_pred = BatchNormPredictor(device=device)
    # batchnorm_pred.load_data(BATCHNORM_PATH)
    # batchnorm_pred.train("predicator_model_batchnorm.th",
    #                      batch_size=512,
    #                      num_epoch=200,
    #                      hooks=[lambda : print("error on test set:", batchnorm_pred.test_set_error())])

def load_model():
    linear_pred = LinearPredictor()
    linear_pred.load_model("./model/predictor_model_linear.th")

    conv_pred = Conv2DPredictor(modulo=modulo)
    conv_pred.load_model("./model/predictor_model_conv2d.th")

    maxpool_pred = MaxPoolingPredictor() 
    maxpool_pred.load_model("./model/predictor_model_maxpool.th")

    return linear_pred, conv_pred, maxpool_pred

if __name__ == '__main__':
    # load_model()
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=["conv2d", "mm", "batchnorm", "maxpool2d", "bmm"])
    parser.add_argument("--nomodulo", action='store_true')
    args = parser.parse_args()
    train(args)
    # conv_pred = Conv2DPredictor()
    # conv_pred.load_model("predictor_model_conv2d.th")
    # conv_pred.load_data(CONV2D_PATH)
    # print(error)
    # conv_pred.xgb_fit()
    # conv_pred.train_set_error()
    # pass
