from python import Python

def read_image(image_path: String) -> PythonObject:
    var cv = Python.import_module("cv2")
    var np = Python.import_module("numpy") 
    var image = cv.imread(image_path)
    print(image.shape)
    nimage = cv.resize(image, (100,100))
    nimage = nimage.astype(np.float32)
    print(nimage.shape)
    nimage /= 255.0

    # print(nimage)
    return nimage

def read_image_yolo(image_path: String) -> PythonObject:
    var cv = Python.import_module("cv2")
    var np = Python.import_module("numpy") 
    var image = cv.imread(image_path, cv.COLOR_BGR2RGB)
    nimage = cv.resize(image, (100,100))
    nimage = nimage.astype(np.float32)
    print(nimage.shape)
    nimage /= 255.0



