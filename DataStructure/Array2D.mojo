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