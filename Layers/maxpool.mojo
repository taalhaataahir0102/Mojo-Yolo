from tensor import Tensor, TensorShape, randn
from time import now
import math
from utils.index import Index

fn maxpool(img: Tensor[DType.float32], pool_size_x:Int, pool_size_y:Int) raises -> Tensor[DType.float32]:
    var N:Int = img.shape()[0]
    var H:Int = img.shape()[1]
    var W:Int = img.shape()[2]
    var C:Int = img.shape()[3]

    var out_H:Int = (H - pool_size_x) // pool_size_x + 1
    var out_W:Int = (H - pool_size_y) // pool_size_y + 1

    var out = Tensor[DType.float32] (N,out_H,out_W,C)
    for n in range(N):
        for h in range(out_H):
            for w in range(out_W):
                for c in range(C):
                    var max_val:Float32 = 0.0
                    for i in range(pool_size_x):
                        for j in range(pool_size_y):
                            max_val = math.max(max_val, img[n, h * pool_size_x + i, w * pool_size_y + j, c])
                    out[Index(n, h, w, c)] = max_val
    
    return out