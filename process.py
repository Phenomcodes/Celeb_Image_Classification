import cv2
import numpy as np
import pywt
import json


face_cascade = cv2.CascadeClassifier("./cascade_algos/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./cascade_algos/haarcascade_eye.xml")


def get_faces(img_path):
    """Accepts a PIL image and returns the face region (if detected with 2+ eyes)."""
    # Convert to NumPy array (if not already)
    if not isinstance(img_path, np.ndarray):
        image = np.array(img_path)

    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
        else:
            print("Image not accepted")

# FEATURE EXTRACTION FUNCTION
def w2d(img, mode = 'haar', level = 1):
    # ✅ If it's a PIL Image, convert to NumPy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #convert to float and normalize
    gray = np.float32(gray)/ 225.0

    # Apply 2D Discrete Wavelet Transform
    coeffs = pywt.wavedec2(gray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    #reconstruction
    img_reconstructed = pywt.waverec2(coeffs_H, mode)
    img_reconstructed *= 255
    img_reconstructed = np.uint8(img_reconstructed)

    return img_reconstructed

#Convert number to name
with open("./artifacts/class_dictionary.json", "r") as f:
    class_dict = json.load(f)

def name_from_number(number):
    for k, v in class_dict.items():
        if number == v:
            return k

