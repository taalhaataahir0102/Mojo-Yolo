from tensor import Tensor
from python import Python as py
from utils.index import Index
from Datastructure.Array2D import Array4D

fn save_image(inout img:Array4D ,image:PythonObject) raises:
    # var input = py.import_module("builtins").input

    for i in range(img.dim0):
        for j in range(img.dim1):
            for k in range(img.dim2):
                for l in range(img.dim3):
                    img[i,j,k,l] = image[i][j][k][l].to_float64().cast[DType.float32]()

fn save_image_mask(inout img:Array4D ,image:PythonObject) raises:
    # var input = py.import_module("builtins").input
    # img.from_numpy(image)
    for i in range(img.dim0):
        for j in range(img.dim1):
            for k in range(img.dim2):
                for l in range(img.dim3):
                    img[i,j,k,l] = image[i][j][k].to_float64().cast[DType.float32]()