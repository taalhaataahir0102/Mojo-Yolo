from max.graph import Graph, TensorType, Type
from max import engine
from tensor import Tensor, TensorShape, randn
from time import now
from python import Python
from algorithm import parallelize, vectorize
from algorithm import Static2DTileUnitFunc as Tile2DFunc

alias type = DType.float32
alias nelts = simdwidthof[type]() * 4

struct Matrix(CollectionElement):
    var data: DTypePointer[type]
    var rows: Int
    var cols: Int

    # Initialize zeroeing all values
    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    fn __copyinit__(inout self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = DTypePointer[type].alloc(self.rows * self.cols)
        for i in range(self.rows * self.cols):
            self.data[i] = existing.data[i]
    
    fn __moveinit__(inout self, owned existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.data = existing.data

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(inout self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)


# Perform 2D tiling on the iteration space defined by end_x and end_y
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul
fn matmul_tiled(inout C: Matrix, A: Matrix, B: Matrix):
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
        tile[calc_tile, 100, 100](C.cols, B.rows)
    parallelize[calc_row](C.rows, C.rows)




def main():
    graph = Graph(in_types=List[Type](TensorType(DType.float32, 500,500), TensorType(DType.float32, 500,500)))
    out = graph[0] @ graph[1]
    graph.output(out)
    graph.verify()

    session = engine.InferenceSession()
    model = session.load(graph)

    var input0 = randn[DType.float32]((500, 500))
    var input1 = randn[DType.float32]((500, 500))
    var start = now()
    for i in range(1000):
        var ret = model.execute("input0", input0, "input1", input1)
    var end = now()

    var execution_time_seconds :  Float32 = (end-start) / 1000000000
    print("MAX GRAPH API:",execution_time_seconds)

    var np = Python.import_module("numpy")
    array1 = np.random.rand(500, 500)
    array2 = np.random.rand(500, 500)
    var start1 = now()
    for i in range(1000):
        var result = np.dot(array1, array2)
    var end1 = now()

    var execution_time_seconds1 :  Float32 = (end1-start1) / 1000000000
    print("NUMPY:",execution_time_seconds1)


    # var matrix1 = Matrix(500,500)
    # var matrix2 = Matrix(500,500)
    # var out_mat = Matrix(500,500)
    # var start2 = now()
    # for i in range(1000):
    #     matmul_tiled(out_mat, matrix1, matrix2)
    # var end2 = now()
    # var execution_time_seconds2 :  Float32 = (end2-start2) / 1000000000
    # print("MOJO:",execution_time_seconds2)


