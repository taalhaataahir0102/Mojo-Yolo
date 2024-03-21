from DataStructure.Array2D import Array3D
from math import ceil, floor
from python import Python as py

# fn get_subarray(img: Array2D, start_row: Int, start_col: Int, row_size: Int, col_size: Int) -> Array2D:
#         var subarray = Array2D(row_size, col_size)
#         for i in range(row_size):
#             for j in range(col_size):
#                 if (start_row + i < img.dim0) and (start_col + j < img.dim1):
#                     subarray.__setitem__(i, j, img.__getitem__(start_row + i, start_col + j))
#                 else:
#                     subarray.__setitem__(i,j,-999)
#         return subarray

# fn maxi(img: Array2D) -> Float32:
#     var m:Float32 = img.__getitem__(0,0)
#     for i in range(img.dim0):
#         for j in range(img.dim1):
#             if img.__getitem__(i,j) > m:
#                 m = img.__getitem__(i,j)
#     return m

# fn maxpool(img: Array2D, pool:Int) -> Array2D:
#     var new_img = Array2D(int(ceil(img.dim0/pool)), int(ceil(img.dim1/pool)))
#     for i in range(0, img.dim0, pool):
#         for j in range(0, img.dim1, pool):
#             var subarray:Array2D = get_subarray(img,i ,j,pool,pool)
#             var m:Float32 = maxi(subarray)
#             var r:Int = i // pool
#             var c:Int = j // pool
#             new_img.__setitem__(r, c, m)
#     return new_img

# fn DepthMaxpooling(img: DynamicVector[Array2D], pool:Int) raises -> DynamicVector[Array2D]:
#     var size:Int = len(img)
#     var output = DynamicVector[Array2D] (capacity=size)
#     for i in range(size):
#         output[i] = maxpool(img[i],pool)

#     return output


fn get_subarray(img: Array3D, start_row: Int, start_col: Int, row_size: Int, col_size: Int) -> Array3D:
        var subarray = Array3D(img.dim0,row_size, col_size)
        for i in range(subarray.dim0):
            for j in range(row_size):
                for k in range(col_size):
                    if (start_row + j < img.dim1) and (start_col + k < img.dim2):
                        subarray.__setitem__(i, j, k, img.__getitem__(i, start_row + j, start_col + k))
                    else:
                        subarray.__setitem__(i,j,k,-999)
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
    var new_img = Array3D(img.dim0,int(ceil(img.dim1/pool)), int(ceil(img.dim2/pool)))

    for j in range(0, img.dim1, pool):
        for k in range(0, img.dim2, pool):
            var subarray:Array3D = get_subarray(img,j ,k,pool,pool)
            # subarray.__printarray__()
            # var user_input: PythonObject = input("Enter to continue")
            for i in range(new_img.dim0):
                var m:Float32 = maxi(subarray,i)
                var r:Int = j // pool
                var c:Int = k // pool
                new_img.__setitem__(i, r, c, m)            
    return new_img

