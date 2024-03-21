from DataStructure.Array2D import Array2D, Array3D
from python import Python as py
import math

# fn get_subarray(img: Array2D, start_row: Int, start_col: Int, row_size: Int, col_size: Int, filter:Array2D) -> Array2D:
#     var subarray = Array2D(row_size, col_size)
#     for i in range(row_size):
#         for j in range(col_size):
#             subarray.__setitem__(i, j, img.__getitem__(start_row + i, start_col + j))
#     return subarray
    
# fn multiply_and_sum(a: Array2D, b: Array2D) -> Float32:
#     var sum:Float32 = 0
#     for i in range(a.dim0):
#         for j in range(a.dim1):
#             sum += a.__getitem__(i, j) * b.__getitem__(i, j)
#     return sum

# fn pading(img: Array2D, pad:Int) -> Array2D:
#     var new_img = Array2D(img.dim0+2*pad, img.dim1+2*pad)
#     for i in range(new_img.dim0):
#         for j in range(new_img.dim1):
#             if i < pad or i >= img.dim0 + pad or j < pad or j >= img.dim1 + pad:
#                 new_img.__setitem__(i,j,0)
#     for i in range(img.dim0):
#         for j in range(img.dim1):
#             new_img.__setitem__(i+pad, j+pad, img.__getitem__(i,j))
#     return new_img

# fn Conv2D(img: Array2D, borrowed filter: Array2D, pad:Int, stride: Int, bias: Float32) raises -> Array2D:
    
#     var h:Int = img.dim0
#     var w:Int = img.dim1
#     var f:Int = filter.dim0
#     var n_h:Int = int((h-f+(2*pad))/stride)+1
#     var n_w:Int = int((w-f+(2*pad))/stride)+1
#     var result = Array2D(n_h,n_w)
#     var new_img:Array2D = pading(img,pad)
#     for i in range(n_h):
#         for j in range(n_w):
#             var subarray:Array2D = get_subarray(new_img,i*stride ,j*stride,f,f, filter)
#             result.__setitem__(i,j,multiply_and_sum(subarray,filter) + bias)
#     return result

# fn multifilterConv2D(img: Array2D, fltrs:DynamicVector[Array2D], pad:Int, stride: Int, bias: DynamicVector[Float32]) raises -> DynamicVector[Array2D]:
#     var size:Int = len(fltrs)
#     var output = DynamicVector[Array2D] (capacity=size)
#     for i in range(size):
#         output[i] = Conv2D(img, fltrs[i],pad, stride, bias[i])

#     return output


fn pading(img: Array3D, pad:Int) -> Array3D:
    var new_img = Array3D(img.dim0,img.dim1+2*pad, img.dim2+2*pad)
    for i in range(new_img.dim0):
        for j in range(new_img.dim1):
            for k in range(new_img.dim2):
                if j < pad or j >= img.dim0 + pad or k < pad or k >= img.dim1 + pad:
                    new_img.__setitem__(i,j,k,0)
    for i in range(img.dim0):
        for j in range(img.dim1):
            for k in range(img.dim2):
                new_img.__setitem__(i,j+pad, k+pad, img.__getitem__(i,j,k))
    return new_img

fn get_subarray(img: Array3D, start_row: Int, start_col: Int, row_size: Int, col_size: Int) -> Array3D:
    var subarray = Array3D(img.dim0,row_size, col_size)
    for i in range(subarray.dim0):
        for j in range(row_size):
            for k in range(col_size):
                subarray.__setitem__(i, j, k, img.__getitem__(i,start_row + j, start_col + k))
    return subarray

fn multiply_and_sum(img: Array3D, filter: Array3D) -> Float32:
    var sum:Float32 = 0
    for i in range(img.dim0):
        for j in range(img.dim1):
            for k in range(img.dim2):
                sum += img.__getitem__(i, j, k) * filter.__getitem__(i, j, k)
    return sum

fn Conv2D(img: Array3D, filter: Array3D, pad:Int, stride: Int, bias: DynamicVector[Float32]) raises -> Array3D:
    var total_filter:Int = int(filter.dim0/img.dim0)
    print("total_filter:",total_filter)
    if filter.dim0/img.dim0 != int(filter.dim0/img.dim0):
        raise Error ("Invalid filter shape")
    if len(bias) != total_filter:
        raise Error ("Invalid bias shape")
    var input = py.import_module("builtins").input
    var h:Int = img.dim1
    var w:Int = img.dim2
    var f:Int = filter.dim1
    var n_h:Int = int((h-f+(2*pad))/stride)+1
    var n_w:Int = int((w-f+(2*pad))/stride)+1
    var result = Array3D(total_filter,n_h,n_w)
    var new_img:Array3D = pading(img,pad)
    for i in range(total_filter):
        var new_filter = Array3D(img.dim0,filter.dim1,filter.dim2)
        for x in range(new_filter.dim0):
            for y in range(new_filter.dim1):
                for z in range(new_filter.dim2):
                    new_filter.__setitem__(x,y,z,filter.__getitem__((i*new_filter.dim0)+x,y,z))
        for j in range(n_h):
            for k in range(n_w):
                var subarray:Array3D = get_subarray(new_img,j*stride ,k*stride,f,f)
                # var user_input: PythonObject = input("Enter to continue")
                result.__setitem__(i,j,k, multiply_and_sum(subarray,new_filter) + bias[i])
    return result