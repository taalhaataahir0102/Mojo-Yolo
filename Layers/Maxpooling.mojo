from DataStructure.Array2D import Array3D
from math import ceil, floor
from python import Python as py

fn get_subarray(img: Array3D, start_row: Int, start_col: Int, row_size: Int, col_size: Int, new_img: Array3D) -> Array3D:
        var subarray = Array3D(new_img.dim0,row_size, col_size)
        for i in range(subarray.dim0):
            for j in range(row_size):
                for k in range(col_size):
                    # if (start_row + j < new_img.dim1) and (start_col + k < new_img.dim2):
                    subarray.__setitem__(i, j, k, img.__getitem__(i, start_row + j, start_col + k))
                    # else:
                    #     subarray.__setitem__(i,j,k,-999)
        return subarray

fn maxi(img: Array3D, depth:Int) -> Float32:
    var m:Float32 = img.__getitem__(depth,0,0)
    for i in range(img.dim1):
        for j in range(img.dim2):
            if img.__getitem__(depth,i,j) > m:
                m = img.__getitem__(depth, i,j)
    return m

fn maxpool(img: Array3D, pool:Int) raises -> Array3D:
    var input = py.import_module("builtins").input
    var new_img = Array3D(img.dim0,int(floor(img.dim1/pool)), int(floor(img.dim2/pool)))

    for j in range(0, img.dim1 - pool + 1, pool):
        for k in range(0, img.dim2 - pool + 1, pool):
            # print("j:", j, "k:", k, "img.dim2:", img.dim2, "pool:", pool)
            var subarray:Array3D = get_subarray(img,j ,k,pool,pool, new_img)
            # subarray.__printarray__()
            # var user_input: PythonObject = input("Enter to continue")
            for i in range(new_img.dim0):
                var m:Float32 = maxi(subarray,i)
                var r:Int = j // pool
                var c:Int = k // pool
                new_img.__setitem__(i, r, c, m)
    return new_img

