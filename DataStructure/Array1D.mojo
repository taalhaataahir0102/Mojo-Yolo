struct Array1D(CollectionElement):
    var data: Pointer[Float32]
    var dim: Int

    fn __init__(inout self, dim: Int):
        self.dim = dim
        self.data = Pointer[Float32].alloc(dim)
    
    fn __copyinit__(inout self, other: Array1D):
        self.dim = other.dim
        self.data = Pointer[Float32].alloc(self.dim)
        for i in range(self.dim):
            self.data.store(i, other.data.load(i))
    
    fn __moveinit__(inout self, owned existing: Array1D):
        self.dim = existing.dim
        self.data = existing.data

    fn __getitem__(borrowed self, i: Int) -> Float32:
        return self.data.load(i)

    fn __setitem__(inout self, i: Int, value: Float32):
        self.data.store(i, value)

    fn __del__(owned self):
        self.data.free()
    
    fn __printarray__(self):
        for i in range(self.dim):
            if i < 5:
                print(self.__getitem__(i), end = ",")
            elif i >= self.dim - 5:
                print(self.__getitem__(i), end = ",")
            else:
                print("...")
        print()