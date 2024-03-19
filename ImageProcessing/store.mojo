from DataStructure.Array2D import Array2D

fn save_image(inout img:Array2D, image:PythonObject) raises:
    for i in range(img.dim0):
        for j in range(img.dim1):
            img.__setitem__(i,j,image[i][j].__index__())

