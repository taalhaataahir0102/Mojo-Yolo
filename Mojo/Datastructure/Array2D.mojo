from tensor import Tensor
from python import Python

alias type = DType.float32
alias nelts = simdwidthof[type]() * 2

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

    fn __len__(self) -> Int:
        return self.rows*self.cols

    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var np_arr = np.zeros((self.rows,self.cols), dtype=np.float32)
        var npArrayPtr = DTypePointer[DType.float32](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, DType.float32.value, `>>`]
        ](
            SIMD[DType.index,1](np_arr.__array_interface__['data'][0].__index__()).value
        )
    )
        memcpy(npArrayPtr, self.data, len(self))
        return np_arr

    fn total_dims(self) -> Int:
        return self.rows * self.cols

    fn create(self, dim0: Int, dim1: Int, _matPtr: Pointer[Float32]) -> Self:
        var instance: Self = Self(dim0, dim1)
        instance.data = _matPtr
        return instance

    fn from_numpy(self, np_array: PythonObject) raises -> Self:
        var npArrayPtr = DTypePointer[DType.float32](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, DType.float32.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        var dim0:Int = 0
        var dim1:Int = 0
        if len(np_array.shape) == 3:
            dim0 = int(np_array.shape[0])
            dim1 = int(np_array.shape[1])
        else:
            dim0 = int(np_array.shape[0])
            dim1 = int(np_array.shape[1])

        var _matPtr = Pointer[Float32].alloc(dim0*dim1)
        memcpy(_matPtr, npArrayPtr, dim0*dim1)
        return Self.create(self, dim0, dim1, _matPtr)

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
        memset_zero(self.data, dim0 * dim1 * dim2 * dim3)
    
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
            print(new_dim0 * new_dim1 * new_dim2 * new_dim3, "!=", self.dim0 * self.dim1 * self.dim2 * self.dim3)
            raise Error("Reshape not allowed")
        self.dim0 = new_dim0
        self.dim1 = new_dim1
        self.dim2 = new_dim2
        self.dim3 = new_dim3
    
    fn __reshape2__(inout self, new_rows: Int, new_cols: Int) raises -> Matrix:
        if new_rows * new_cols != self.dim0 * self.dim1 * self.dim2 * self.dim3:
            print(new_rows * new_cols, "!=", self.dim0 * self.dim1 * self.dim2 * self.dim3)
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

    fn total_dims(self) -> Int:
        return self.dim0 * self.dim1 * self.dim2 * self.dim3
    
    fn __flatten__(self) -> Tensor[DType.float32]:
        var flat = Tensor[DType.float32] (self.dim0 * self.dim1 * self.dim2 * self.dim3)
        for i in range(self.dim0 * self.dim1 * self.dim2 * self.dim3):
            flat[i] = self.data.load(i)
        return flat

    fn create(self, dim0: Int, dim1: Int, dim2: Int, dim3: Int, _matPtr: Pointer[Float32]) -> Self:
        var instance: Self = Self(dim0, dim1, dim2, dim3)
        instance.data = _matPtr
        return instance

    
    fn from_numpy(self, np_array: PythonObject) raises -> Self:
        var npArrayPtr = DTypePointer[DType.float32](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, DType.float32.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        var dim0:Int = 0
        var dim1:Int = 0
        var dim2:Int = 0
        var dim3:Int = 0
        if len(np_array.shape) == 3:
            dim0 = int(np_array.shape[0])
            dim1 = int(np_array.shape[1])
            dim2 = int(np_array.shape[2])
            dim3 = 1
        else:
            dim0 = int(np_array.shape[0])
            dim1 = int(np_array.shape[1])
            dim2 = int(np_array.shape[2])
            dim3 = int(np_array.shape[3])
        var _matPtr = Pointer[Float32].alloc(dim0*dim1*dim2*dim3)
        memcpy(_matPtr, npArrayPtr, dim0*dim1*dim2*dim3)
        return Self.create(self, dim0, dim1, dim2, dim3, _matPtr)

struct Array1D(CollectionElement):
    var data: Pointer[Float32]
    var dim0: Int

    fn __init__(inout self, dim0: Int):
        self.dim0 = dim0
        self.data = Pointer[Float32].alloc(dim0)
        memset_zero(self.data, dim0)
    
    fn __copyinit__(inout self, other: Array1D):
        self.dim0 = other.dim0
       
        self.data = Pointer[Float32].alloc(self.dim0)
        for i in range(self.dim0):
            self.data.store(i, other.data.load(i))
    
    fn __moveinit__(inout self, owned existing: Array1D):
        self.dim0 = existing.dim0
        self.data = existing.data

    fn __getitem__(borrowed self, i: Int) -> Float32:
        return self.data.load(i)

    fn __setitem__(inout self, i: Int, value: Float32):
        self.data.store(i, value)

    fn __del__(owned self):
        self.data.free()

    fn __printarray__(self):
        for i in range(self.dim0):
            print(self.__getitem__(i), end = ",")

    fn __shape__(self):
        print("shape:", self.dim0)

    fn total_dims(self) -> Int:
        return self.dim0

    fn create(self, dim0: Int, _matPtr: Pointer[Float32]) -> Self:
        var instance: Self = Self(dim0)
        instance.data = _matPtr
        return instance
    
    fn from_numpy(self, np_array: PythonObject) raises -> Self:
        var npArrayPtr = DTypePointer[DType.float32](
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<`, DType.float32.value, `>>`]
        ](
            SIMD[DType.index,1](np_array.__array_interface__['data'][0].__index__()).value
        )
    )
        var dim0:Int = 0

        dim0 = int(np_array.shape[0])
        
        var _matPtr = Pointer[Float32].alloc(dim0)
        memcpy(_matPtr, npArrayPtr, dim0)
        return Self.create(self, dim0, _matPtr)