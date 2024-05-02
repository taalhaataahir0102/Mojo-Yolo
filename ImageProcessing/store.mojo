from DataStructure.Array2D import Array2D, Array3D
from python import Python as py

fn save_image(inout img:Array2D, image:PythonObject) raises:
    var input = py.import_module("builtins").input
    for i in range(img.dim0):
        for j in range(img.dim1):
            img.__setitem__(i,j,image[i][j].to_float64().cast[DType.float32]())
            # print(image[i][j].__index__())
            # var user_input: PythonObject = input("Enter to continue")

fn save_image(inout img:Array3D, inout image:PythonObject) raises:
    var input = py.import_module("builtins").input
    image = image.transpose()
    for i in range(img.dim0):
        for j in range(img.dim1):
            for k in range(img.dim2):
                # print(image[i][j][k].to_float64().cast[DType.float32]())
                # var user_input: PythonObject = input("Enter to continue")
                img.__setitem__(i,j,k,image[i][j][k].to_float64().cast[DType.float32]())

