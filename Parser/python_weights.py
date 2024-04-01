import h5py
import numpy as np

def test(s):
    return float(s)
    
def read2_hdf5(path):
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f:  # open file
        f.visit(keys.append)  # append all keys to list
        for key in keys:
            if ':' in key:  # contains data if ':' in key
                if 'conv2d' in key:  # check if it's a Conv2D layer
                    # Transpose the Conv2D kernel before storing it
                    weights[f[key].name] = f[key][()].transpose()
                else:
                    weights[f[key].name] = f[key][()]
    return weights

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


def print_array(array):
    if array.ndim == 1:
        print(', '.join(map(str, array)))
    elif array.ndim == 2:
        for row in array:
            print(', '.join(map(str, row)))
    elif array.ndim == 4:
        for dim1 in array:
            for dim2 in dim1:
                for dim3 in dim2:
                    print(', '.join(map(str, dim3)))
    else:
        print("Unsupported shape.")

def print_array(array, file):
    if array.ndim == 1:
        file.write('\n'.join(map(str, array)) + "\n")
    elif array.ndim == 2:
        for row in array:
            file.write('\n'.join(map(str, row)) + "\n")
    elif array.ndim == 4:
        for dim1 in array:
            for dim2 in dim1:
                for dim3 in dim2:
                    file.write('\n'.join(map(str, dim3)) + "\n")
    else:
        file.write("Unsupported shape.\n")

def main():
    path_to_hdf5 = "/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5"
    weights = read2_hdf5(path_to_hdf5)

    with open("/home/talha/Desktop/mojo/yolo/Parser/weights.txt", "w") as file:
        for key, value in weights.items():
            file.write(f"Key: {key}, Shape: {value.shape}\n")
            print_array(value, file)
            file.write("\n")


if __name__ == "__main__":
    main()
