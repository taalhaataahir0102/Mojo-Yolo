from tensor import Tensor
alias type = DType.float32
alias nelts = simdwidthof[type]() * 2

struct Array2D(CollectionElement):
    var data: Pointer[Float32]
    var dim0: Int
    var dim1: Int

    fn __init__(inout self, dim0: Int, dim1: Int):
        self.dim0 = dim0
        self.dim1 = dim1
        self.data = Pointer[Float32].alloc(dim0 * dim1)
    
    fn __copyinit__(inout self, other: Array2D):
        self.dim0 = other.dim0
        self.dim1 = other.dim1
        self.data = Pointer[Float32].alloc(self.dim0 * self.dim1)
        for i in range(self.dim0 * self.dim1):
            self.data.store(i, other.data.load(i))
    
    fn __moveinit__(inout self, owned existing: Array2D):
        self.dim0 = existing.dim0
        self.dim1 = existing.dim1
        self.data = existing.data

    fn __getitem__(borrowed self, i: Int, j: Int) -> Float32:
        return self.data.load(i * self.dim1 + j)

    fn __setitem__(inout self, i: Int, j: Int, value: Float32):
        self.data.store(i * self.dim1 + j, value)

    fn __del__(owned self):
        self.data.free()

    fn __printarray__(self):
        for i in range(self.dim0):
            for j in range(self.dim1):
                if (j < 8 or j > self.dim1- 8) and (i < 8 or i > self.dim0 - 8):
                    print(self.__getitem__(i,j), end = ",")
                elif j == 9 and (i < 9 or i > self.dim0 - 9):
                    print(",...", end = ",")
                else:
                    continue
            if i < 9 or i > self.dim0 - 9:
                print()

    fn save_to_file(borrowed self, path: String) raises:
        
        with open(path, "w") as out:
            for i in range(self.dim0):
                for j in range(self.dim1):
                    var value = self.__getitem__(i, j)
                    out.write(str(value) + "\n")
            out.close()

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

    fn flatten(self) -> Tensor[DType.float32]:
        var flat = Tensor[DType.float32] (self.rows * self.cols)
        for i in range(self.rows * self.cols):
            flat[i] = self.data.load(i)
        return flat

    fn __printarray__(self):
        print("shape:",self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                print(self.__getitem__(i,j), end = ",")
            print()

    fn __reshape__(inout self, new_rows:Int, new_cols:Int) raises:
        if self.rows * self.cols != new_rows * new_cols:
            raise Error("Not allowed to rshape")
        else:
            self.rows = new_rows
            self.cols = new_cols

    # fn __reshape2__(inout self, new_dim0: Int, new_dim1: Int, new_dim2: Int, new_dim3: Int) raises -> Array4D:
    #     if new_dim0 * new_dim1 * new_dim2 * new_dim3 != self.rows * self.cols:
    #         raise Error("Not allowed to rshape")
        
    #     var array4D = Array4D(new_dim0, new_dim1, new_dim2, new_dim3)
    #     for i in range(new_dim0):
    #         for j in range(new_dim1):
    #             for k in range(new_dim2):
    #                 for l in range(new_dim3):
    #                     array4D[i, j, k, l] = self.data.load(i * new_dim1 * new_dim2 * new_dim3 + j * new_dim2 * new_dim3 + k * new_dim3 + l)
    #     return array4D

    fn __reshape2__(inout self, new_dim0: Int, new_dim1: Int, new_dim2: Int, new_dim3: Int) raises -> Array4D:
        if new_dim0 * new_dim1 * new_dim2 * new_dim3 != self.rows * self.cols:
            raise Error("Not allowed to reshape")
        
        var array4D = Array4D(new_dim0, new_dim1, new_dim2, new_dim3)
        for i in range(new_dim0):
            var i_offset = i * new_dim1 * new_dim2 * new_dim3
            for j in range(new_dim1):
                var j_offset = j * new_dim2 * new_dim3
                for k in range(new_dim2):
                    var k_offset = k * new_dim3
                    for l in range(new_dim3):
                        array4D[i, j, k, l] = self.data.load(i_offset + j_offset + k_offset + l)
        return array4D
    
    fn __shape__(self):
        print("shape:", self.rows,self.cols)

struct Array3D(CollectionElement):
    var data: Pointer[Float32]
    var dim0: Int
    var dim1: Int
    var dim2: Int

    fn __init__(inout self, dim0: Int, dim1: Int, dim2: Int):
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.data = Pointer[Float32].alloc(dim0 * dim1 * dim2)
    
    fn __copyinit__(inout self, other: Array3D):
        self.dim0 = other.dim0
        self.dim1 = other.dim1
        self.dim2 = other.dim2
        self.data = Pointer[Float32].alloc(self.dim0 * self.dim1 * self.dim2)
        for i in range(self.dim0 * self.dim1 * self.dim2):
            self.data.store(i, other.data.load(i))
    
    fn __moveinit__(inout self, owned existing: Array3D):
        self.dim0 = existing.dim0
        self.dim1 = existing.dim1
        self.dim2 = existing.dim2
        self.data = existing.data

    fn __getitem__(borrowed self, i: Int, j: Int, k: Int) -> Float32:
        return self.data.load(i * self.dim1 * self.dim2 + j * self.dim2 + k)

    fn __setitem__(inout self, i: Int, j: Int, k: Int, value: Float32):
        self.data.store(i * self.dim1 * self.dim2 + j * self.dim2 + k, value)

    fn __del__(owned self):
        self.data.free()

    
    fn __printarray__(self):
        for i in range(self.dim0):
            print(i)
            for j in range(self.dim1):
                for k in range(self.dim2):
                    if (k < 8 or k > self.dim1- 8) and (j < 8 or j > self.dim0 - 8):
                        print(self.__getitem__(i,j,k), end = ",")
                    elif k == 9 and (j < 9 or j > self.dim0 - 9):
                        print("...", end = ",")
                    else:
                        continue
                if j < 9 or j > self.dim0 - 9:
                    print()


struct Array4D(CollectionElement):
    var data: Pointer[Float32]
    var dim0: Int
    var dim1: Int
    var dim2: Int
    var dim3: Int

    fn __init__(inout self, dim0: Int, dim1: Int, dim2: Int, dim3: Int):
        self.dim0 = dim0
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.data = Pointer[Float32].alloc(dim0 * dim1 * dim2 * dim3)
    
    fn __copyinit__(inout self, other: Array4D):
        self.dim0 = other.dim0
        self.dim1 = other.dim1
        self.dim2 = other.dim2
        self.dim3 = other.dim3
        self.data = Pointer[Float32].alloc(self.dim0 * self.dim1 * self.dim2 * self.dim3)
        for i in range(self.dim0 * self.dim1 * self.dim2 * self.dim3):
            self.data.store(i, other.data.load(i))
    
    fn __moveinit__(inout self, owned existing: Array4D):
        self.dim0 = existing.dim0
        self.dim1 = existing.dim1
        self.dim2 = existing.dim2
        self.dim3 = existing.dim3
        self.data = existing.data

    fn __getitem__(borrowed self, i: Int, j: Int, k: Int, l: Int) -> Float32:
        return self.data.load(i * self.dim1 * self.dim2 * self.dim3 + j * self.dim2 * self.dim3 + k * self.dim3 + l)

    fn __setitem__(inout self, i: Int, j: Int, k: Int, l: Int, value: Float32):
        self.data.store(i * self.dim1 * self.dim2 * self.dim3 + j * self.dim2 * self.dim3 + k * self.dim3 + l, value)

    fn __del__(owned self):
        self.data.free()

    fn __reshape__(inout self, new_dim0: Int, new_dim1: Int, new_dim2: Int, new_dim3: Int) raises:
        if new_dim0 * new_dim1 * new_dim2 * new_dim3 != self.dim0 * self.dim1 * self.dim2 * self.dim3:
            raise Error("Reshape not allowed")
        self.dim0 = new_dim0
        self.dim1 = new_dim1
        self.dim2 = new_dim2
        self.dim3 = new_dim3
    
    fn __reshape2__(inout self, new_rows: Int, new_cols: Int) raises -> Matrix:
        if new_rows * new_cols != self.dim0 * self.dim1 * self.dim2 * self.dim3:
            raise Error("Reshape not allowed")
        var matrix = Matrix(new_rows, new_cols)
        for i in range(new_rows):
            for j in range(new_cols):
                matrix[i, j] = self.data.load(i * new_cols + j)
        return matrix

    fn __printarray__(self):
        print("shape:", self.dim0,self.dim1,self.dim2,self.dim3)
        for i in range(self.dim0):
            for j in range(self.dim1):
                for k in range(self.dim2):
                    for l in range(self.dim3):    
                        print(self.__getitem__(i,j,k,l), end = ",")
                    print()
                print()
    fn __shape__(self):
        print("shape:", self.dim0,self.dim1,self.dim2,self.dim3)
    
    fn __flatten__(self) -> Tensor[DType.float32]:
        var flat = Tensor[DType.float32] (self.dim0 * self.dim1 * self.dim2 * self.dim3)
        for i in range(self.dim0 * self.dim1 * self.dim2 * self.dim3):
            flat[i] = self.data.load(i)
        return flat