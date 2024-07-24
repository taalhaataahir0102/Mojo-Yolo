from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import warnings
import logging
import os
import torch
import cv2
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def resize_image(image, size=(150, 150)):
    return image.resize(size, Image.BILINEAR)

def pre_processing(image_path, image_processor):
    image = Image.open(image_path)
    image = resize_image(image, size=(150, 150))
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].numpy()

class CustomYolosObjectDetectionOutput:
    def __init__(self, logits, pred_boxes, last_hidden_state):
        self.logits = logits
        self.pred_boxes = pred_boxes
        self.last_hidden_state = last_hidden_state
        self.loss = None
        self.loss_dict = None
        self.auxiliary_outputs = None
        self.hidden_states = None
        self.attentions = None

def draw_boxes(image, results, labels_map, threshold=0.8):
    image = np.array(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score >= threshold:
            box = [round(i, 2) for i in box.tolist()]
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            color = (0, 0, 255)  # Red color in BGR
            thickness = 2

            # Draw rectangle
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

            # Prepare label text
            label_text = f"{labels_map[label.item()]}: {round(score.item(), 3)}"

            # Set the font scale and thickness
            font_scale = 0.5
            font_thickness = 1

            # Get the text size
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Draw the background rectangle for the text
            image = cv2.rectangle(image, start_point, (start_point[0] + text_width, start_point[1] - text_height - 4), color, -1)

            # Put the label text
            image = cv2.putText(image, label_text, (start_point[0], start_point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return image

def post_processing(logits, pred_boxes, last_hidden_state, img, image_processor, model):
    image = Image.open(img)
    image = resize_image(image, size=(150, 150))
    custom_yolos_output = CustomYolosObjectDetectionOutput(
        logits=torch.from_numpy(logits),
        pred_boxes=torch.from_numpy(pred_boxes),
        last_hidden_state=torch.from_numpy(last_hidden_state)
    )
    
    results = image_processor.post_process_object_detection(custom_yolos_output, threshold=0.8, target_sizes=[[150,150]])[0]

    labels_map = model.config.id2label
    image_with_boxes = draw_boxes(image, results, labels_map)

    output_image_path = 'output_stream/' + img
    # Save the image with bounding boxes
    cv2.imwrite(output_image_path, cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    print("output path:", output_image_path)
    return image_with_boxes