from DataStructure.Array2D import Array2D, Array3D
from python import Python as py
import math
from Activations.Relu import relu

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
        print(len(bias), "!=", total_filter)
        raise Error ("Invalid bias shape")
    var input = py.import_module("builtins").input
    var h:Int = img.dim1
    var w:Int = img.dim2
    var f:Int = filter.dim1
    var n_h:Int = int((h+(2*pad)-f)/stride) + 1
    var n_w:Int = int((w+(2*pad)-f)/stride) + 1

    # print("n_h, n_w", n_h, n_w)
    # print("Pad:", pad, "stride:", stride, "h:", h, "w:", w, "f:", f)
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
                result.__setitem__(i,j,k, multiply_and_sum(subarray,new_filter) + bias[i])
                if i == 0 and (j == 0 or j == 1) and k ==0 and stride == 2:
                    print("yes:")
                    subarray.__printarray__()
                    new_filter.__printarray__()
                    print(bias[i])
                    print(result.__getitem__(i,j,k))
                    var user_input: PythonObject = input("Enter to continue")
    
    var ans:Array3D = relu(result)
    return ans