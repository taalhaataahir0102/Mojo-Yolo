from python import Python

def read_image(image_path: String) -> PythonObject:
    var cv = Python.import_module("cv2")
    var np = Python.import_module("numpy") 
    var image = cv.imread(image_path)
    nimage = cv.resize(image, (80,80))
    nimage = nimage.astype(np.float32)
    nimage /= 255.0
    image = np.expand_dims(nimage, axis=0)
    print("image.shape", image.shape)
    return image

def read_image_mask(image_path: String) -> PythonObject:
    var cv = Python.import_module("cv2")
    var np = Python.import_module("numpy") 
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    nimage = cv.resize(image, (144,144))
    nimage = nimage.astype(np.float32)
    nimage /= 255.0
    image = np.expand_dims(nimage, axis=0)
    # image1 = np.expand_dims(image, axis=-1)
    
    return image