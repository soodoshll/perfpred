import sqlite3
from timeit import default_timer
from unittest import defaultTestLoader
import torch
from torch import nn
from tqdm import trange, tqdm
import random
import time
import numpy as np
from functools import partial
import xgboost as xgb
from torchmetrics import MeanAbsolutePercentageError
import glob, pickle

LINEAR_PATH = glob.glob("matmul_data_*.data")
CONV2D_PATH_SQL = ["./habitat-data/conv2d/conv2d-RTX2080Ti-0.sqlite", "./habitat-data/conv2d/conv2d-RTX2080Ti-1.sqlite"]
CONV2D_PATH = glob.glob("conv_data_*.data")
MAXPOOLING_PATH = ['./pool_10000.npz']

device = torch.device('cuda')

def make_mlp(device, input_dim, hidden_layers=[1024] * 8):
    layers = []
    last = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, 1, bias=False))
    model = nn.Sequential(*layers)
    model.to(device)    
    return model

def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target))    

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
        model.to(device)
        dataloader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        
        loss_fn = MAPELoss 
        optim = torch.optim.Adam(
                model.parameters(), 
                lr=1e-4, 
                # weight_decay=1e-4
                )
        
        for epoch_idx in trange(num_epoch):
            for data in dataloader:
                inputs = data[:, :-1].to(device)
                labels = data[:, -1].to(device)
                optim.zero_grad(set_to_none=True)
                out = model(inputs)[:, 0]
                loss = loss_fn(out, labels)
                loss.backward()
                optim.step()

            if epoch_idx % 10 == 0:
                for hook in hooks:
                    hook()
                print(f"{epoch_idx} : {loss.detach().cpu().numpy() : .5f} | {labels[0] :.2f} {out[0] :.2f}")
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
        inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = (inputs - self.avgs[:-1]) / self.stds[:-1]
        out = self.model(inputs)
        # print(out, self.stds[-1], self.avgs[-1])
        out = out[0] * self.stds[-1] + self.avgs[-1]
        return out.detach().numpy()


class LinearPredictor(Predictor):
    def __init__(self):
        self.feature_name = ['bias', 'batch', 'in_features', 'out_features', 'is_forward']
        self.model = make_mlp(device, len(self.feature_name))
        self.device = device

    def load_data(self, filenames):
        rows = []
        for fn in filenames:
            with open(fn, "rb") as f:
                while 1:
                    try:
                        objs = pickle.load(f)
                        for obj in objs:
                            # print(obj)
                            dur_forward, dur_backward, batch_size, in_features, out_features = obj
                            rows.append(
                                (1, batch_size, in_features, out_features, 1, dur_forward)
                            )
                    except (EOFError):
                        break
        self.raw_dataset = torch.tensor(rows, dtype=torch.float32)
        print(self.raw_dataset[0])
        # self.raw_dataset[:, -1] /= 1000
        # print(torch.sum(self.raw_dataset[:, -1] == 0))
        print(self.raw_dataset[0])
        print("datasize:", len(rows))
        

        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.avgs[-1] = 0
        self.stds[-1] = 1

        ####
        self.avgs[0] = 0
        self.stds[0] = 1
        self.avgs[-2] = 0
        self.stds[-2] = 1
        ####

        self.dataset = (self.raw_dataset - self.avgs) / self.stds
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

class Conv2DPredictor(Predictor):
    def __init__(self):
        self.feature_name = ['bias', 'batch', 'image_size', 'in_channels', 'out_channels', 'kernel_size',
        'stride', 'padding', 'is_forward']
        self.model = make_mlp(device, len(self.feature_name))
        self.device = device

        self.xgb_r = xgb.XGBRegressor(objective ='reg:squarederror',
                  n_estimators = 200, seed = 123)
    
    def load_data_mix(self, sql_filenames, py_filenames):
        rows = [] 
        for fn in py_filenames:
            with open(fn, "rb") as f:
                while 1:
                    try:
                        objs = pickle.load(f)
                        for obj in objs:
                            # print(obj)
                            dur_forward, dur_backward, dx, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = obj
                            rows.append(
                                (1, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, dur_forward)
                            )
                            # rows.append(
                                # (1, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 0, dur_backward)
                            # )
                    except (EOFError):
                        break        
        for path in sql_filenames:
            conn = sqlite3.connect(path)
            c = conn.cursor()
            cursor = c.execute("SELECT * from recordings")
            rows.extend([row[1:] for row in cursor if row[1] == 0])                        
        
        self.raw_dataset = torch.tensor(rows, dtype=torch.float32)
        self.raw_dataset[:, 0] = 0
        print(self.raw_dataset[0])
        # self.raw_dataset[:, -1] /= 1000
        # print(torch.sum(self.raw_dataset[:, -1] == 0))
        print(self.raw_dataset[0])
        print("datasize:", len(rows))
        

        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.avgs[-1] = 0
        self.stds[-1] = 1

        ####
        self.avgs[0] = 0
        self.stds[0] = 1
        self.avgs[-2] = 0
        self.stds[-2] = 1
        ####

        self.dataset = (self.raw_dataset - self.avgs) / self.stds
        print("avg:", self.avgs)
        print("std:", self.stds)  

        train_set_size = int(self.dataset.shape[0] * 0.8)
        test_set_size = self.dataset.shape[0] - train_set_size
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_set_size, test_set_size]) 


    def load_data(self, filenames):
        rows = []
        for fn in filenames:
            with open(fn, "rb") as f:
                while 1:
                    try:
                        objs = pickle.load(f)
                        for obj in objs:
                            # print(obj)
                            dur_forward, dur_backward, dx, batch_size, kernel_size, image_size, in_channels, out_channels, stride, padding = obj
                            rows.append(
                                (1, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 1, dur_forward)
                            )
                            # rows.append(
                                # (1, batch_size, image_size, in_channels, out_channels, kernel_size, stride, padding, 0, dur_backward)
                            # )
                    except (EOFError):
                        break
        self.raw_dataset = torch.tensor(rows, dtype=torch.float32)
        print(self.raw_dataset[0])
        # self.raw_dataset[:, -1] /= 1000
        # print(torch.sum(self.raw_dataset[:, -1] == 0))
        print(self.raw_dataset[0])
        print("datasize:", len(rows))
        

        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.avgs[-1] = 0
        self.stds[-1] = 1

        ####
        self.avgs[0] = 0
        self.stds[0] = 1
        self.avgs[-2] = 0
        self.stds[-2] = 1
        ####

        self.dataset = (self.raw_dataset - self.avgs) / self.stds
        print("avg:", self.avgs)
        print("std:", self.stds)  

        train_set_size = int(self.dataset.shape[0] * 0.8)
        test_set_size = self.dataset.shape[0] - train_set_size
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_set_size, test_set_size]) 
            

    def train_set_error(self, batch_size=1000):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True) 
        errors = []
        for data in dataloader:
            inputs = data[:, :-1].to(device)
            labels = data[:, -1].to(device)
            out = self.model(inputs)
            pred = out[:, 0] * self.stds[-1] + self.avgs[-1]
            truth = labels * self.stds[-1] + self.avgs[-1]
            error = torch.abs(pred - truth) / truth
            errors.append(error)
        return torch.mean(torch.concat(errors))

    def test_set_error(self, batch_size=1000):
        dataloader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=True) 
        errors = []
        for data in dataloader:
            inputs = data[:, :-1].to(device)
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
    def __init__(self):
        # self.params = (batch_size, kernel_size, image_size, channels, stride)
        self.feature_name = ['batch_size', 'kernel_size', 'image_size', 'channels', 'stride', 'is_forward']
        self.model = make_mlp(device, len(self.feature_name))
        self.device = device        

    def load_data(self, paths):
        rows = []
        for path in paths:
            data = np.load(path)
            data_fw = data['pool_fw']
            rows_fw = np.concatenate((data_fw[:, :-1], np.ones((data_fw.shape[0], 1)), data_fw[:, -1:]), axis=-1)
            rows.append(rows_fw)
            data_bw = data['pool_bw']
            rows_bw = np.concatenate((data_bw[:, :-1], np.zeros((data_bw.shape[0], 1)), data_bw[:, -1:]), axis=-1)
            rows.append(rows_bw)
        rows = np.concatenate(rows)
        rows = rows.astype('float32')

        self.raw_dataset = torch.tensor(rows)
        print("datasize:", len(rows))
        
        self.avgs = torch.mean(self.raw_dataset, axis=0)
        self.stds = torch.std(self.raw_dataset, axis=0)
        self.dataset = (self.raw_dataset - self.avgs) / self.stds
        print("avg:", self.avgs)
        print("std:", self.stds)   
        

def train():
    linear_pred = LinearPredictor()
    linear_pred.load_data(LINEAR_PATH)
    linear_pred.train('predictor_model_linear.th', 
                    batch_size=512,
                    num_epoch=80, 
                    hooks=[lambda : print("error on test set:", conv_pred.test_set_error().detach().cpu().numpy())])
    return

    conv_pred = Conv2DPredictor()
    # conv_pred.load_data_mix(sql_filenames=CONV2D_PATH_SQL, py_filenames=CONV2D_PATH)
    conv_pred.load_data_mix(sql_filenames=[], py_filenames=CONV2D_PATH)
    # return 
    conv_pred.train("predictor_model_conv2d.th",
                    batch_size=512,
                    num_epoch=200, 
                    hooks=[lambda : print("error on test set:", conv_pred.test_set_error().detach().cpu().numpy())])
    # error = conv_pred.train_set_error()
    # maxpool_pred = MaxPoolingPredictor()
    # maxpool_pred.load_data(MAXPOOLING_PATH)
    # maxpool_pred.train("predicator_model_maxpool.th", num_epoch=500)

def load_model():
    linear_pred = LinearPredictor()
    linear_pred.load_model("predictor_model_linear.th")

    conv_pred = Conv2DPredictor()
    conv_pred.load_model("predictor_model_conv2d.th")

    maxpool_pred = MaxPoolingPredictor() 
    maxpool_pred.load_model("predicator_model_maxpool.th")

    return linear_pred, conv_pred, maxpool_pred

if __name__ == '__main__':
    # load_model()
    train()
    # conv_pred = Conv2DPredictor()
    # conv_pred.load_model("predictor_model_conv2d.th")
    # conv_pred.load_data(CONV2D_PATH)
    # conv_pred.xgb_fit()
    # conv_pred.train_set_error()
    # pass
