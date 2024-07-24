import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided

def read_hdf5(path):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                if 'conv2d' in key: # check if it's a Conv2D layer
                    # Transpose the Conv2D kernel before storing it
                    weights[f[key].name] = f[key][()].shape
                else:
                    weights[f[key].name] = f[key][()].shape
    # print(weights)
    return weights

def read2_hdf5(path, key_name):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                if 'conv2d' in key: # check if it's a Conv2D layer
                    # Transpose the Conv2D kernel before storing it
                    weights[f[key].name] = f[key][()]
                else:
                    weights[f[key].name] = f[key][()]
    # print(weights)
    return weights[key_name]

def test(s):
    return float(s)

def strided_conv(Z,weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    Ns, Hs, Ws, Cs = Z.strides
    inner_dim = K*K*C_in
    A=as_strided(Z, shape=(N, H-K+1, W-K+1, K, K, C_in),
                strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1,inner_dim)
    w = weight.reshape(-1, C_out)
    return A,w

def result_reshape(Z,weight,res):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    return res.reshape (N, H-K+1, W-K+1, C_out)

def main():
    a = read2_hdf5("model_weights.h5", '/conv2d_30/conv2d_30/kernel:0')
    print("helo")
    print(a)


if __name__ == "__main__":
    main()