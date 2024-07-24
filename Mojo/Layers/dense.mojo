from tensor import Tensor
from python import Python
from python import Python as py
from Datastructure.Array2D import Array4D, Matrix
from random import rand
from algorithm import parallelize, vectorize
from time import now
from algorithm import Static2DTileUnitFunc as Tile2DFunc

alias nelts = simdwidthof[DType.float32] () * 128


# Perform 2D tiling on the iteration space defined by end_x and end_y
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

fn dense(inout C: Matrix, A: Matrix, B: Matrix, bias: List[Float32],layer:Int):
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
        if layer == 0:
            tile[calc_tile, 1, 1](C.cols, B.rows)

        for n in range(C.cols):
            print(C[m,n])
            C[m, n] += bias[n]
            if C[m,n] <=0:
                C[m,n] = 0