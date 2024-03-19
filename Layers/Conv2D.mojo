from DataStructure.Array2D import Array2D
from python import Python as py

fn get_subarray(img: Array2D, start_row: Int, start_col: Int, row_size: Int, col_size: Int) -> Array2D:
        var subarray = Array2D(row_size, col_size)
        for i in range(row_size):
            for j in range(col_size):
                subarray.__setitem__(i, j, img.__getitem__(start_row + i, start_col + j))
        return subarray
    
fn multiply_and_sum(a: Array2D, b: Array2D) -> Float32:
    var sum:Float32 = 0
    for i in range(a.dim0):
        for j in range(a.dim1):
            sum += a.__getitem__(i, j) * b.__getitem__(i, j)
    return sum

fn pading(inout img: Array2D, pad:Int):
    var new_img = Array2D(img.dim0+2*pad, img.dim1+2*pad)
    for i in range(new_img.dim0):
        for j in range(new_img.dim1):
            if i < pad or i >= img.dim0 + pad or j < pad or j >= img.dim1 + pad:
                new_img.__setitem__(i,j,0)
    for i in range(img.dim0):
        for j in range(img.dim1):
            new_img.__setitem__(i+pad, j+pad, img.__getitem__(i,j))
    img  = new_img

fn Conv2D(inout img: Array2D, filter: Array2D, pad:Int, stride: Int, bias: Float32) raises -> Array2D:
    var input = py.import_module("builtins").input
    var h:Int = img.dim0
    var w:Int = img.dim1
    var f:Int = filter.dim0
    var n_h:Int = int((h-f+(2*pad))/stride)+1
    var n_w:Int = int((w-f+(2*pad))/stride)+1
    var result = Array2D(n_h,n_w)
    pading(img,pad)
    for i in range(n_h):
        for j in range(n_w):
            var subarray:Array2D = get_subarray(img,i*stride ,j*stride,f,f)
            result.__setitem__(i,j,multiply_and_sum(subarray,filter) + bias)
    return result
