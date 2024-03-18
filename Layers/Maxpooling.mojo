from DataStructure.TensorVector import Array2D
from math import ceil, floor

fn get_subarray(img: Array2D, start_row: Int, start_col: Int, row_size: Int, col_size: Int) -> Array2D:
        var subarray = Array2D(row_size, col_size)
        for i in range(row_size):
            for j in range(col_size):
                if (start_row + i < img.dim0) and (start_col + j < img.dim1):
                    subarray.__setitem__(i, j, img.__getitem__(start_row + i, start_col + j))
                else:
                    subarray.__setitem__(i,j,-999)
        return subarray

fn maxi(img: Array2D) -> Float32:
    var m:Float32 = img.__getitem__(0,0)
    for i in range(img.dim0):
        for j in range(img.dim1):
            if img.__getitem__(i,j) > m:
                m = img.__getitem__(i,j)
    return m

fn maxpool(inout img: Array2D, pool:Int):
    var new_img = Array2D(int(ceil(img.dim0/pool)), int(ceil(img.dim1/pool)))
    for i in range(0, img.dim0, pool):
        for j in range(0, img.dim1, pool):
            var subarray:Array2D = get_subarray(img,i ,j,pool,pool)
            var m:Float32 = maxi(subarray)
            var r:Int = i // pool
            var c:Int = j // pool
            new_img.__setitem__(r, c, m)
    img = new_img



