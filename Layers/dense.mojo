from tensor import Tensor, TensorShape, randn
from python import Python
from python import Python as py
from utils.index import Index
from max.graph import Graph, TensorType, Type, ops
from max import engine


fn dense(img: Tensor[DType.float32], weight: Tensor[DType.float32], bias: Tensor[DType.float32], layer: String) raises -> Tensor[DType.float32]:
    var img_r = img.shape() [0]
    var img_c = img.shape() [1]
    var weight_r = weight.shape() [0]
    var weight_c = weight.shape() [1]

    if layer == "sigmoid":
        var graph = Graph(in_types=List[Type](TensorType(DType.float32, img_r, img_c), TensorType(DType.float32, weight_r, weight_c), TensorType(DType.float32, bias.shape()[0])))
        var out = graph[0] @ graph[1]
        var out1 = out + graph[2]
        var out2 = ops.sigmoid(out1)
        graph.output(out2)
        graph.verify()
        var session = engine.InferenceSession()
        var model = session.load(graph)
        var out_names = model.get_model_output_names()
        var ret = model.execute("input0", img, "input1", weight, "input2", bias)
        var t1 = ret.get[DType.float32] (out_names[0])
        return t1
    elif layer == "relu":
        var graph = Graph(in_types=List[Type](TensorType(DType.float32, img_r, img_c), TensorType(DType.float32, weight_r, weight_c), TensorType(DType.float32, bias.shape()[0])))
        var out = graph[0] @ graph[1]
        var out1 = out + graph[2]
        var out2 = ops.relu(out1)
        graph.output(out2)
        graph.verify()
        var session = engine.InferenceSession()
        var model = session.load(graph)
        var out_names = model.get_model_output_names()
        var ret = model.execute("input0", img, "input1", weight, "input2", bias)
        var t1 = ret.get[DType.float32] (out_names[0])
        return t1

    else:
        var graph = Graph(in_types=List[Type](TensorType(DType.float32, img_r, img_c), TensorType(DType.float32, weight_r, weight_c), TensorType(DType.float32, bias.shape()[0])))
        var out = graph[0] @ graph[1]
        var out1 = out + graph[2]
        graph.output(out1)
        graph.verify()
        var session = engine.InferenceSession()
        var model = session.load(graph)
        var out_names = model.get_model_output_names()
        var ret = model.execute("input0", img, "input1", weight, "input2", bias)
        var t1 = ret.get[DType.float32] (out_names[0])
        return t1