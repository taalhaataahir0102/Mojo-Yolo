from Datastructure.Array2D import Array4D, Matrix
from time import now
import math

fn maxpool(img: Array4D, pool_size_x:Int, pool_size_y:Int) raises -> Array4D:
    var N:Int = img.dim0
    var H:Int = img.dim1
    var W:Int = img.dim2
    var C:Int = img.dim3

    var out_H:Int = (H - pool_size_x) // pool_size_x + 1
    var out_W:Int = (H - pool_size_y) // pool_size_y + 1

    var out:Array4D = Array4D(N,out_H,out_W,C)
    for n in range(N):
        for h in range(out_H):
            for w in range(out_W):
                for c in range(C):
                    var max_val:Float32 = 0.0
                    for i in range(pool_size_x):
                        for j in range(pool_size_y):
                            max_val = max(max_val, img[n, h * pool_size_x + i, w * pool_size_y + j, c])
                    out[n, h, w, c] = max_val
    
    return out