from python import Python
from python import Python as py
from DataStructure.Array2D import Array2D, Array3D

fn layers_manipulation(s:String, l:Int) -> DynamicVector[String]:
    var k = DynamicVector[String] (capacity = l)
    var str_len:Int = s.__len__()
    var idx:Int = 0
    var found:Bool = False
    var start:String = ''
    while(idx < str_len):
        if s[idx] == "'" and found == False:
            found = True
            idx+=1
        elif found == True and s[idx] != "'":
            start += s[idx]
            idx+=1
        elif s[idx] == "'" and found == True:
            idx +=1
            k.append(start)
            found = False
            start = ''
        else:
            idx+=1
        
    return k

fn shape_manipulation(s:String, l:Int) -> DynamicVector[String]:
    var k = DynamicVector[String] (capacity = l)
    var str_len:Int = s.__len__()
    var idx:Int = 0
    var found:Bool = False
    var start:String = ''
    var first:Bool = False
    while(idx < str_len):
        if s[idx] == "(" and found == False and first == True:
            found = True
            idx+=1
        elif s[idx] == "(" and found == False and first == False:
            first = True
            idx+=1
            continue
        elif found == True and s[idx] != ")":
            start += s[idx]
            idx+=1
        elif s[idx] == ")" and found == True:
            idx +=1
            k.append(start)
            found = False
            start = ''
        else:
            idx+=1
        
    return k

fn split_string(s:String, split:String) -> DynamicVector[String]:
    var start:String = ''
    var k = DynamicVector[String] ()
    for i in range(s.__len__()):
        if s[i] == split:
            k.append(start)
            start = ''
        else:
            start += s[i]
    if start == '':
        return k
    else:
        k.append(start)
        return k

fn get_total_conv_layers(layers: DynamicVector[String])->Int:
    var ans:Int = 0
    for i in range(len(layers)):
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            ans+=1
    return ans

fn get_total_dense_layers(layers: DynamicVector[String])->Int:
    var ans:Int = 0
    for i in range(len(layers)):
        if layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
            ans+=1
    return ans

fn get_total_bias_layers(layers: DynamicVector[String])->Int:
    var ans:Int = 0
    for i in range(len(layers)):
        if layers[i].__contains__("bias"):
            ans+=1
    return ans

fn string_vec_to_int (inout s:DynamicVector[String]) raises -> DynamicVector[Int]:
    var new_vec = DynamicVector[Int] ()
    for i in range(len(s)):
        s[i] = s[i].strip()
        var n:Int = atol(s[i])
        new_vec.append(n)
    return new_vec

fn pop_front (v:DynamicVector[String])-> DynamicVector[String]:
    var new_v = DynamicVector[String] ()
    for i in range(1,len(v),1):
        new_v.append(v[i])
    return new_v

fn extract_number(s:String) raises -> Float32:
    Python.add_to_path("/home/talha/Desktop/mojo/yolo/Parser/")
    var mypython = Python.import_module("weights")
    var numeric: String = ""
    for i in range(len(s)):
        if s[i].__contains__(".")  or s[i].__contains__("-") or s[i].__contains__("0") or s[i].__contains__("1") or s[i].__contains__("2") or s[i].__contains__("3") or s[i].__contains__("4") or s[i].__contains__("5") or s[i].__contains__("6") or s[i].__contains__("7") or s[i].__contains__("8") or s[i].__contains__("9") or s[i].__contains__("e"):
            numeric += s[i]
    var a: PythonObject = mypython.test(numeric)
    var b:Float32 = a.to_float64().cast[DType.float32]()
    return b

fn collect_weights() raises:
    var input = py.import_module("builtins").input
    Python.add_to_path("/home/talha/Desktop/mojo/yolo/Parser/")
    var mypython = Python.import_module("weights")
    var my_dict: PythonObject = mypython.read_hdf5("/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5")

    var keys = my_dict.keys()

    var s: String = keys.__str__()
    var l:Int = keys.__len__()
    
    var layers = DynamicVector[String] (capacity = l)
    layers = layers_manipulation(s,l)


    var items = my_dict.values()

    var s2: String = items.__str__()
    var l2:Int = items.__len__()
    
    var total_conv_layers = get_total_conv_layers(layers)
    var total_dense_layers = get_total_dense_layers(layers)
    var total_bias_layers = get_total_bias_layers(layers)

    var conv2D_weights = DynamicVector[Array3D] (capacity = total_conv_layers)
    var dense_weights = DynamicVector[Array2D] (capacity = total_dense_layers)
    var baises = DynamicVector[DynamicVector[Float32]] (capacity = total_bias_layers)

    var shapes = DynamicVector[String] (capacity = l)
    shapes = shape_manipulation(s2,l2)

    for i in range(len(shapes)):
        
        var split_values = split_string(shapes[i], ",")
        var int_sizes = string_vec_to_int(split_values)
        
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            var new_filter = Array3D(int_sizes[0]*int_sizes[1], int_sizes[2], int_sizes[3])
            conv2D_weights.append(new_filter)
        elif layers[i].__contains__("bias"):
            var new_bias = DynamicVector[Float32] (capacity = int_sizes[0])
            baises.append(new_bias)
        elif layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
            var new_dense = Array2D(int_sizes[0], int_sizes[1])
            dense_weights.append(new_dense)

    print("LAYERS:")
    for i in range(len(layers)):
        print(layers[i])

    print("CONV2D LAYERS", len(conv2D_weights))
    for i in range(len(conv2D_weights)):
        print(conv2D_weights[i].dim0,conv2D_weights[i].dim1,conv2D_weights[i].dim2)
    
    print("DENSE LAYERS", len(dense_weights))
    for i in range(len(dense_weights)):
        print(dense_weights[i].dim0,dense_weights[i].dim1)

    print("BIAS LAYERS", len(baises))
    for i in range(len(baises)):
        print(baises[i].capacity)

    
    print("==============HERE==============")
    var my_dict2: PythonObject = mypython.read2_hdf5("/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5")
    var values = my_dict2.values().__str__()

    var splitted_values: DynamicVector[String] = values.split("array")
    var new_splitted_values = DynamicVector[String] (capacity = len(layers))
    new_splitted_values = pop_front(splitted_values)

    var conv_index:Int = 0
    var dense_index:Int = 0
    var bias_index:Int = 0

    for i in range(len(new_splitted_values)):
        # print(new_splitted_values[i])
        var solo_weights: DynamicVector[String] = new_splitted_values[i].split(",")
        var loop_size:Int = 0
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            loop_size = conv2D_weights[conv_index].dim0 * conv2D_weights[conv_index].dim1 * conv2D_weights[conv_index].dim2
            conv_index +=1
        elif layers[i].__contains__("bias"):
            loop_size = baises[bias_index].capacity
            bias_index +=1
        elif layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
            loop_size = dense_weights[dense_index].dim0 * dense_weights[dense_index].dim1
            dense_index +=1
        

        for j in range(loop_size):
            var num:Float32 = extract_number(solo_weights[j])
            if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
                var index0:Int = j // (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)  # Compute the index along the first dimension
                var index1:Int = (j % (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)) // conv2D_weights[conv_index-1].dim2  # Compute the index along the second dimension
                var index2:Int = (j % (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)) % conv2D_weights[conv_index-1].dim2
                conv2D_weights[conv_index-1].__setitem__(index0,index1,index2,num)
                # conv2D_weights[conv_index-1].__printarray__()
                # var user_input: PythonObject = input("Enter to continue")
            elif layers[i].__contains__("bias"):
                # print("YES:", bias_index-1, j , num)
                baises[bias_index-1].append(num)
                
            elif layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
                var index0:Int = j // dense_weights[dense_index-1].dim1  # Compute the index along the first dimension
                var index1:Int = j % dense_weights[dense_index-1].dim1
                dense_weights[dense_index-1].__setitem__(index0,index1,num)

    
    print("WEIGHTS")
    # for i in range(len(conv2D_weights)):
    #     conv2D_weights[i].__printarray__()
    
    # for i in range(len(baises)):
    #     for j in range(len(baises[i])):
    #         print(baises[i][j])

    for i in range(len(dense_weights)):
        dense_weights[i].__printarray__()

fn collect_conv2d_weights(inout conv2D_weights:DynamicVector[Array3D]) raises:
    var input = py.import_module("builtins").input
    Python.add_to_path("/home/talha/Desktop/mojo/yolo/Parser/")
    var mypython = Python.import_module("weights")
    var my_dict: PythonObject = mypython.read_hdf5("/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5")

    var keys = my_dict.keys()
    var s: String = keys.__str__()
    var l:Int = keys.__len__()
    
    var layers = DynamicVector[String] (capacity = l)
    layers = layers_manipulation(s,l)

    var items = my_dict.values()
    var s2: String = items.__str__()
    var l2:Int = items.__len__()
    
    var total_conv_layers = get_total_conv_layers(layers)
    conv2D_weights = DynamicVector[Array3D] (capacity = total_conv_layers)

    var shapes = DynamicVector[String] (capacity = l)
    shapes = shape_manipulation(s2,l2)

    for i in range(len(shapes)):
        var split_values = split_string(shapes[i], ",")
        var int_sizes = string_vec_to_int(split_values)
        
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            var new_filter = Array3D(int_sizes[0]*int_sizes[1], int_sizes[2], int_sizes[3])
            conv2D_weights.append(new_filter)

    var my_dict2: PythonObject = mypython.read2_hdf5("/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5")
    var values = my_dict2.values().__str__()

    var splitted_values: DynamicVector[String] = values.split("array")
    var new_splitted_values = DynamicVector[String] (capacity = len(layers))
    new_splitted_values = pop_front(splitted_values)

    var conv_index:Int = 0

    for i in range(len(new_splitted_values)):
        # print(new_splitted_values[i])
        var solo_weights: DynamicVector[String] = new_splitted_values[i].split(",")
        var loop_size:Int = 0
        if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
            loop_size = conv2D_weights[conv_index].dim0 * conv2D_weights[conv_index].dim1 * conv2D_weights[conv_index].dim2
            conv_index +=1

        for j in range(loop_size):
            var num:Float32 = extract_number(solo_weights[j])
            if layers[i].__contains__("conv2d") and layers[i].__contains__("kernel"):
                var index0:Int = j // (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)  # Compute the index along the first dimension
                var index1:Int = (j % (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)) // conv2D_weights[conv_index-1].dim2  # Compute the index along the second dimension
                var index2:Int = (j % (conv2D_weights[conv_index-1].dim1 * conv2D_weights[conv_index-1].dim2)) % conv2D_weights[conv_index-1].dim2
                conv2D_weights[conv_index-1].__setitem__(index0,index1,index2,num)


fn collect_dense_weights(inout dense_weights:DynamicVector[Array2D]) raises:
    var input = py.import_module("builtins").input
    Python.add_to_path("/home/talha/Desktop/mojo/yolo/Parser/")
    var mypython = Python.import_module("weights")
    var my_dict: PythonObject = mypython.read_hdf5("/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5")

    var keys = my_dict.keys()
    var s: String = keys.__str__()
    var l:Int = keys.__len__()
    
    var layers = DynamicVector[String] (capacity = l)
    layers = layers_manipulation(s,l)

    var items = my_dict.values()
    var s2: String = items.__str__()
    var l2:Int = items.__len__()
    
    var total_dense_layers = get_total_dense_layers(layers)
    dense_weights = DynamicVector[Array2D] (capacity = total_dense_layers)
    
    var shapes = DynamicVector[String] (capacity = l)
    shapes = shape_manipulation(s2,l2)

    for i in range(len(shapes)):
        var split_values = split_string(shapes[i], ",")
        var int_sizes = string_vec_to_int(split_values)
        
        if layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
            var new_dense = Array2D(int_sizes[0], int_sizes[1])
            dense_weights.append(new_dense)

    var my_dict2: PythonObject = mypython.read2_hdf5("/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5")
    var values = my_dict2.values().__str__()

    var splitted_values: DynamicVector[String] = values.split("array")
    var new_splitted_values = DynamicVector[String] (capacity = len(layers))
    new_splitted_values = pop_front(splitted_values)

    var dense_index:Int = 0
    print(my_dict2.values())
    print("=====================")
    print(values)

    for i in range(len(new_splitted_values)):        
        var solo_weights: DynamicVector[String] = new_splitted_values[i].split(",")
        var loop_size:Int = 0
        if layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
            loop_size = dense_weights[dense_index].dim0 * dense_weights[dense_index].dim1
            dense_index +=1
        for j in range(loop_size):
            # print("DONE:",solo_weights[j])
            var num:Float32 = extract_number(solo_weights[j])
            if layers[i].__contains__("dense") and layers[i].__contains__("kernel"):
                var index0:Int = j // dense_weights[dense_index-1].dim1  # Compute the index along the first dimension
                var index1:Int = j % dense_weights[dense_index-1].dim1
                dense_weights[dense_index-1].__setitem__(index0,index1,num)

fn collect_baises(inout baises: DynamicVector[DynamicVector[Float32]]) raises:
    var input = py.import_module("builtins").input
    Python.add_to_path("/home/talha/Desktop/mojo/yolo/Parser/")
    var mypython = Python.import_module("weights")
    var my_dict: PythonObject = mypython.read_hdf5("/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5")

    var keys = my_dict.keys()
    var s: String = keys.__str__()
    var l:Int = keys.__len__()
    
    var layers = DynamicVector[String] (capacity = l)
    layers = layers_manipulation(s,l)
    var items = my_dict.values()

    var s2: String = items.__str__()
    var l2:Int = items.__len__()
    
    var total_bias_layers = get_total_bias_layers(layers)
    baises = DynamicVector[DynamicVector[Float32]] (capacity = total_bias_layers)

    var shapes = DynamicVector[String] (capacity = l)
    shapes = shape_manipulation(s2,l2)

    for i in range(len(shapes)):     
        var split_values = split_string(shapes[i], ",")
        var int_sizes = string_vec_to_int(split_values)
        if layers[i].__contains__("bias"):
            var new_bias = DynamicVector[Float32] (capacity = int_sizes[0])
            baises.append(new_bias)

    var my_dict2: PythonObject = mypython.read2_hdf5("/home/talha/Desktop/mojo/yolo/Parser/model_weights.h5")
    var values = my_dict2.values().__str__()

    var splitted_values: DynamicVector[String] = values.split("array")
    var new_splitted_values = DynamicVector[String] (capacity = len(layers))
    new_splitted_values = pop_front(splitted_values)

    var conv_index:Int = 0
    var dense_index:Int = 0
    var bias_index:Int = 0

    for i in range(len(new_splitted_values)):
        var solo_weights: DynamicVector[String] = new_splitted_values[i].split(",")
        var loop_size:Int = 0
        if layers[i].__contains__("bias"):
            loop_size = baises[bias_index].capacity
            bias_index +=1

        for j in range(loop_size):
            var num:Float32 = extract_number(solo_weights[j])
            if layers[i].__contains__("bias"):
                baises[bias_index-1].append(num)