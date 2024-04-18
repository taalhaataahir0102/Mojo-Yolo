import os

# Disable GPU usage in TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set TensorFlow logging to warning level only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time


classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}


class_names = {value: key for key, value in classes_num.items()}


def load_and_preprocess_image(image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    original_image = image.copy()
    image_h, image_w = image.shape[0:2]
    image = cv.resize(image, (100, 100))  # Assuming your model was trained with this size
    image = image / 255.0
    return image, original_image, image_h, image_w


def decode_predictions(predictions, image_h, image_w):
    grid_size = 7
    num_boxes = 2
    num_classes = 20
    predictions = predictions.reshape((grid_size, grid_size, 30))

    boxes = []

    for i in range(grid_size):
        for j in range(grid_size):
            for b in range(num_boxes):
                tx, ty, tw, th, confidence = predictions[i, j, b*5:(b+1)*5]
                cx = (j + tx) * image_w / grid_size
                cy = (i + ty) * image_h / grid_size
                w = image_w * np.exp(tw) / grid_size
                h = image_h * np.exp(th) / grid_size
                classes = predictions[i, j, 10:]
                class_id = np.argmax(classes)
                class_score = classes[class_id]
                score = confidence * class_score

                if score > 0.06:  # Confidence threshold
                    class_name = class_names[class_id]
                    boxes.append((cx, cy, w, h, class_name, score))

    return boxes

def draw_boxes(image, boxes):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box in boxes:
        cx, cy, w, h, cls, score = box
        rect = patches.Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(cx, cy, f'Class {cls}, Score: {score:.2f}', bbox=dict(facecolor='red', alpha=0.5))

    # Instead of displaying the image, save it to a file
    plt.axis('off')  # Optional: Turn off the axis
    plt.gca().set_axis_off()  # Optional: Hide the axis
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # Optional: Remove padding and margin around the image
    plt.margins(0,0)  # Optional: Remove margins
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Optional: Remove ticks
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Optional: Remove ticks
    plt.savefig('output.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

class Yolo_Reshape(tf.keras.layers.Layer):
  def __init__(self, target_shape):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    # grids 7x7
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = 20
    # no of bounding boxes per grid
    B = 2

    idx1 = S[0] * S[1] * C
    idx2 = idx1 + S[0] * S[1] * B

    # class probabilities
    class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
    class_probs = K.softmax(class_probs)

    #confidence
    confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
    confs = K.sigmoid(confs)

    # boxes
    boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([class_probs, confs, boxes])
    return outputs

def load_floats_to_array(filepath):
    # Initialize an empty list to hold the float values
    data = []
    
    # Open the file and read line by line
    with open(filepath, 'r') as file:
        for line in file:
            # Convert each line to a float and append to the list
            data.append(float(line.strip()))
    
    # Check if the data length is as expected (1470)
    if len(data) != 1470:
        raise ValueError("The file must contain exactly 1470 float values")
    
    # Convert the list of floats to a numpy array and reshape it
    data_array = np.array(data).reshape(1, 1470)
    
    return data_array


def main():
    final_dense_output = load_floats_to_array('/home/talha/Desktop/mojo/yolo/final_layer.txt')


    start_time = time.time()

    yolo_reshape_layer = Yolo_Reshape(target_shape=(7, 7, 30))

    # Pass the output through the Yolo_Reshape layer
    reshaped_output = yolo_reshape_layer(final_dense_output)

    # Convert the tensor to a numpy array for printing
    reshaped_output_numpy = reshaped_output.numpy()

    # Print the reshaped output


    image_path = '/home/talha/Desktop/mojo/yolo/Example/carbird.jpeg'  # Replace with your image file
    img, original_img, orig_h, orig_w = load_and_preprocess_image(image_path)

    decoded_boxes = decode_predictions(reshaped_output_numpy[0], orig_h, orig_w)

    draw_boxes(original_img, decoded_boxes)


    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time take to draw bounding box:         ", elapsed_time)

if __name__ == "__main__":
    main()
