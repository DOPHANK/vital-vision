import cv2
import os

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def batch_preprocess(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        processed_image = preprocess_image(image_path)
        cv2.imwrite(os.path.join(output_dir, filename), processed_image)
