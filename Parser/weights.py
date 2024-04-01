import h5py
import numpy as np

def read_hdf5(path):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                if 'conv2d' in key: # check if it's a Conv2D layer
                    # Transpose the Conv2D kernel before storing it
                    weights[f[key].name] = f[key][()].transpose().shape
                else:
                    weights[f[key].name] = f[key][()].shape
    return weights

def read2_hdf5(path):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                if 'conv2d' in key: # check if it's a Conv2D layer
                    # Transpose the Conv2D kernel before storing it
                    weights[f[key].name] = f[key][()].transpose()
                else:
                    weights[f[key].name] = f[key][()]
    return weights

def test(s):
    return float(s)