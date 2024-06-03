from tensor import Tensor, TensorShape, randn
from python import Python
from python import Python as py
from utils.index import Index
from max.graph import Graph, TensorType, Type, ops
from max import engine

fn Conv2Dim2col(img: Tensor[DType.float32], inout filter: Tensor[DType.float32], bias: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
    var N:Int = img.shape()[0]
    var H:Int = img.shape()[1]
    var W:Int = img.shape()[2]
    var C_in:Int = img.shape()[3]
    var K:Int = filter.shape()[1]
    var C_out:Int = filter.shape()[3]
    var A = Tensor[DType.float32] ((N * (H - K + 1) * (W - K + 1)) , K * K * C_in)

    
    var index:Int = 0
    for n in range(N):
        for i in range(H - K + 1):
            for j in range(W - K + 1):
                for k1 in range(K):
                    for k2 in range(K):
                        for c_in in range(C_in):
                            A[Index(index, k1 * K * C_in + k2 * C_in + c_in)] = img[n, i + k1, j + k2, c_in]
                index+=1

    var new_shape = TensorShape(K * K * C_in, C_out)
    var wow = filter.reshape(new_shape)

    var graph = Graph(in_types=List[Type](TensorType(DType.float32, (N * (H - K + 1) * (W - K + 1)),K * K * C_in), TensorType(DType.float32, K * K * C_in,C_out), TensorType(DType.float32, bias.shape()[0])))
    var out = graph[0] @ graph[1]
    var out1 = out + graph[2]
    var out2 = ops.relu(out1)
    graph.output(out2)
    graph.verify()

    var session = engine.InferenceSession()
    var model = session.load(graph)

    var out_names = model.get_model_output_names()

    var ret = model.execute("input0", A, "input1", wow, "input2", bias)
    var t1 = ret.get[DType.float32] (out_names[0])

    var out_shape = TensorShape(N, H - K + 1, W - K + 1, C_out)
    var t2 = t1.reshape(out_shape)

    return t2



    
