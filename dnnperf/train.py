import torch
import pickle, time

dataset_file = 'nas_graphs.data'

def load_data_raw(filename):
    with open(filename, 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data

t0 = time.time()
load_data_raw(dataset_file)
dur = time.time() - t0

print(f"dataset loaded, cost {dur} s")