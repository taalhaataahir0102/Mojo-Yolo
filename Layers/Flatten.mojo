from DataStructure.Array2D import Array2D, Array3D

fn flatten(img: Array3D) -> Array2D:
    var flat = Array2D(1,img.dim0*img.dim1*img.dim2)
    var col_index:Int = 0
    for i in range(img.dim0):
        for j in range(img.dim1):
            for k in range(img.dim2):
                flat.__setitem__(0,col_index, img.__getitem__(i,j,k))
                col_index+=1
    return flat

fn flatten2(img: Array3D) -> Array2D:
    var flat = Array2D(1,img.dim0*img.dim1*img.dim2)
    var col_index:Int = 0
    print(img.dim0, img.dim1, img.dim2)
    for k in range(img.dim2):
        for i in range(img.dim0):
            for j in range(img.dim1):
                flat.__setitem__(0,col_index, img.__getitem__(i,j,k))
                col_index+=1
    return flat

fn flatten3(img: Array3D) -> Array2D:
    var flat = Array2D(1,img.dim0*img.dim1*img.dim2)
    var col_index:Int = 0
    for i in range(img.dim2):
        for j in range(img.dim1):
            for k in range(img.dim0):
                flat.__setitem__(0,col_index, img.__getitem__(k,j,i))
                col_index+=1
    return flat