from DataStructure.Array2D import Array3D

fn relu(img:Array3D)-> Array3D:
    var new_img = Array3D(img.dim0,img.dim1,img.dim2)
    for i in range(new_img.dim0):
        for j in range(new_img.dim1):
            for k in range(new_img.dim2):
                if img.__getitem__(i,j,k) <=0:
                    new_img.__setitem__(i,j,k,0)
                else:
                    new_img.__setitem__(i,j,k,img.__getitem__(i,j,k))
    return new_img