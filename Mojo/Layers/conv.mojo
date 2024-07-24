from tensor import Tensor
from python import Python
from python import Python as py
from Datastructure.Array2D import Array4D, Matrix, Array1D
from random import rand
from algorithm import parallelize, vectorize
from time import now
from algorithm import Static2DTileUnitFunc as Tile2DFunc

alias type = DType.float32

alias Alias1 = 1       #64
alias Alias2 = 1      #128


alias nelts = simdwidthof[DType.float32] () * 16


# Perform 2D tiling on the iteration space defined by end_x and end_y
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul
fn matmul_tiled(inout C: Matrix, A: Matrix, B: Matrix, bias: Array1D, act: String = ""):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[dot, nelts, size=tile_x]()
        
        tile[calc_tile, Alias1, Alias2](C.cols, B.rows)

        for n in range(C.cols):
            C[m, n] += bias[n]
            if C[m,n] <=0:
                C[m,n] = 0

    parallelize[calc_row](C.rows, C.rows)
    if act == "relu":
        relu(C)
    elif act == "sigmoid":
        sigmoid(C)

fn relu(inout img: Matrix):
    @parameter
    fn process_pixel(i: Int):
        for j in range(img.cols):
            if img[i,j] < 0:
                img[i,j] = 0
    parallelize[process_pixel] (img.rows)

fn sigmoid(inout img: Matrix):
    @parameter
    fn process_pixel(i: Int):
        for j in range(img.cols):
            img[i, j] = 1 / (1 + math.exp(-img[i, j]))
    parallelize[process_pixel] (img.rows)

fn Conv2Dim2col(img: Array4D, inout filter: Array4D, bias: Array1D, inout time_in_matmul: Float32 ,stride: Int = 1, padding: Int = 0) raises -> Array4D:
    var N: Int = img.dim0
    var H: Int = img.dim1
    var W: Int = img.dim2
    var C_in: Int = img.dim3
    var K: Int = filter.dim0
    var C_out: Int = filter.dim3

    #Add padding to height and width
    var padded_H: Int = H + 2 * padding
    var padded_W: Int = W + 2 * padding

    # Initialize padded image with zeros
    var padded_img: Array4D = Array4D(N, padded_H, padded_W, C_in)

    #Copy the original image into the padded image
    for n in range(N):
        for c in range(C_in):
            for i in range(H):
                for j in range(W):
                    padded_img[n, i + padding, j + padding, c] = img[n, i, j, c]

    var out_H: Int = int((padded_H - K) / stride + 1)
    var out_W: Int = int((padded_W - K) / stride + 1)

    var A = Matrix((N * out_H * out_W), K * K * C_in)

    var index: Int = 0
    for n in range(N):
        for i in range(0, padded_H - K + 1, stride):
            for j in range(0, padded_W - K + 1, stride):
                for k1 in range(K):
                    for k2 in range(K):
                        for c_in in range(C_in):
                            A[index, k1 * K * C_in + k2 * C_in + c_in] = padded_img[n, i + k1, j + k2, c_in]
                index += 1

    var my_filter = filter.__reshape2__(K * K * C_in, C_out)

    var mul = Matrix(A.rows, my_filter.cols)
    
    var start_time = now()
    matmul_tiled(mul, A, my_filter, bias, "relu")
    var end_time = now()
    var elapsed_time = (end_time - start_time)/ 1000000000

    time_in_matmul += elapsed_time

    var out: Array4D = mul.__reshape2__(N, out_H, out_W, C_out)

    return out

fn Conv2D(img: Array4D, inout filter: Array4D, bias: Array1D, stride: Int = 1, padding: String = "valid", activation: String = "relu") raises -> Array4D:
    var N: Int = img.dim0
    var H: Int = img.dim1
    var W: Int = img.dim2
    var C_in: Int = img.dim3
    var K: Int = filter.dim0
    var C_out: Int = filter.dim3
    var pad = 0
    #Add padding to height and width
    if padding == "same":
        pad = 1
    var padded_H: Int = H + 2 * pad
    var padded_W: Int = W + 2 * pad

    # Initialize padded image with zeros
    var padded_img: Array4D = Array4D(N, padded_H, padded_W, C_in)

    #Copy the original image into the padded image
    for n in range(N):
        for c in range(C_in):
            for i in range(H):
                for j in range(W):
                    padded_img[n, i + pad, j + pad, c] = img[n, i, j, c]

    var out_H: Int = int((padded_H - K) / stride + 1)
    var out_W: Int = int((padded_W - K) / stride + 1)

    var A = Matrix((N * out_H * out_W), K * K * C_in)

    var index: Int = 0
    for n in range(N):
        for i in range(0, padded_H - K + 1, stride):
            for j in range(0, padded_W - K + 1, stride):
                for k1 in range(K):
                    for k2 in range(K):
                        for c_in in range(C_in):
                            A[index, k1 * K * C_in + k2 * C_in + c_in] = padded_img[n, i + k1, j + k2, c_in]
                index += 1

    var my_filter = filter.__reshape2__(K * K * C_in, C_out)
    var mul = Matrix(A.rows, my_filter.cols)
    matmul_tiled(mul, A, my_filter, bias, activation)
    var out: Array4D = mul.__reshape2__(N, out_H, out_W, C_out)

    return out

# fn Conv2Dim2col(img: Array4D, filter: Array4D, bias: Array1D, stride:Int = 1) raises -> Array4D:
#     var N: Int = img.dim0
#     var H: Int = img.dim1
#     var W: Int = img.dim2
#     var C_in: Int = img.dim3
#     var K: Int = filter.dim0
#     var C_out: Int = filter.dim3
    
#     var out_H: Int = int((H - K) / stride + 1)
#     var out_W: Int = int((W - K) / stride + 1)
    
#     var A = Matrix((N * out_H * out_W), K * K * C_in)
    
#     var index: Int = 0
#     for n in range(N):
#         for i in range(0, H - K + 1, stride):
#             for j in range(0, W - K + 1, stride):
#                 for k1 in range(K):
#                     for k2 in range(K):
#                         for c_in in range(C_in):
#                             A[index, k1 * K * C_in + k2 * C_in + c_in] = img[n, i + k1, j + k2, c_in]
#                 index += 1
    
#     var my_filter = filter
#     var wow: Matrix = my_filter.__reshape2__(K * K * C_in, C_out)
    
#     var mul = Matrix(A.rows, wow.cols)
#     matmul_tiled(mul, A, wow, bias, "relu")

#     var out: Array4D = mul.__reshape2__(N, out_H, out_W, C_out)
    
#     return out

# fn relu(inout img: Matrix):
#     for i in range(img.rows):
#         for j in range(img.cols):
#             if img[i,j] < 0:
#                 img[i,j] = 0

# fn sigmoid(inout img: Matrix) -> None:
#     for i in range(img.rows):
#         for j in range(img.cols):
#             img[i, j] = 1 / (1 + math.exp(-img[i, j]))

# fn Conv2Dim2col(img: Array4D, filter: Array4D, bias: Array1D) raises -> Array4D:
#     var N:Int = img.dim0
#     var H:Int = img.dim1
#     var W:Int = img.dim2
#     var C_in:Int = img.dim3
#     var K:Int = filter.dim0
#     var C_out:Int = filter.dim3
#     var A = Matrix((N * (H - K + 1) * (W - K + 1)) , K * K * C_in)
    
#     var index:Int = 0
#     for n in range(N):
#         for i in range(H - K + 1):
#             for j in range(W - K + 1):
#                 for k1 in range(K):
#                     for k2 in range(K):
#                         for c_in in range(C_in):
#                             A[index, k1 * K * C_in + k2 * C_in + c_in] = img[n, i + k1, j + k2, c_in]
#                 index+=1
    

#     var my_filter = filter
#     var wow:Matrix = my_filter.__reshape2__(K * K * C_in, C_out)

#     var mul = Matrix(A.rows, wow.cols)
#     matmul_tiled(mul, A, wow, bias, "relu")

#     var out:Array4D = mul.__reshape2__(N, H - K + 1, W - K + 1, C_out)


#     return out





    
