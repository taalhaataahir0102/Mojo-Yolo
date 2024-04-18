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


# [3,2,2,2]
# [6,2,2]