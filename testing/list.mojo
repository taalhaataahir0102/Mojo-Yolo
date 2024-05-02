
fn print_shape(list3D: List[List[List[Int]]]):
    var x = len(list3D)
    var y = len(list3D[0])
    var z = len(list3D[0][0])
    print('Shape:', x, y, z)

fn main():
    var list3D = List[List[List[Int]]] ()

    for x in range(3):
        var list2D = List[List[Int]] ()
        for y in range(3):
            var list1D = List[Int] ()
            for z in range(3):
                list1D.append(x*y+z)
            list2D.append(list1D)
        list3D.append(list2D)

    var x = 1
    var y = 2
    var z = 0
    print(list3D[x][y][z])

    print_shape(list3D)