import cv2
import time
from flask import Flask, render_template, Response, request
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model
from skimage.transform import resize
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

model = load_model("hand_gesture_recognition_6.h5")

input_shape = (120, 120)

def reduce_highlights(image, threshold=220, inpaint_radius=3):

    # create a mask of the highlights
    _, highlight_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # perform inpainting
    final_image = cv2.inpaint(image, highlight_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return final_image

def getPredictedClass(model, img):
    image = img
    image = cv2.resize(image, input_shape)
    gray_image = cv2.resize(image, input_shape)

    gray_image = gray_image.reshape(1, 120, 120, 1)

    # Predict the class
    prediction_array = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction_array, axis=1)

    if predicted_class == 0:
        return "Blank"
    elif predicted_class == 1:
        return "OK"
    elif predicted_class == 2:
        return "Thumbs Up"
    elif predicted_class == 3:
        return "Thumbs Down"
    elif predicted_class == 4:
        return "Punch"
    elif predicted_class == 5:
        return "High Five"

# Function to perform inference and get the predicted gesture
def get_prediction(frame):
    # Perform inference on the frame using the machine learning model
    # Send the frame to the backend for inference and receive the predicted gesture
    gesture = "Predicted Gesture (Dummy Code)"

    return gesture

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        # Code for handling the uploaded image file
        file = request.files['image']
        # Perform prediction on the image file (replace with your actual prediction logic)
        detector = HandDetector(detectionCon=0.2)

        # default crop position
        x = 0
        y = 0

        # Define color thresholds for black and white conversion
        black_threshold = 127
        white_threshold = 128

        predictedClass = ""
        predicted_gesture = "Blank"

        img = plt.imread(file)

        img = cv2.flip(img, 1)
        hands = detector.findHands(img, draw=False)
        hand1 = hands[0] if hands else None

        if hands:
            # Get the hand landmarks
            hand1 = hands[0]  # if there's only one hand

            # Crop the hand from the image
            x, y, w, h = hand1['bbox']
            w = w + 80
            h = h + 80
            x = x - 40
            y = y - 40
            hand_crop = img[y:y + h, x:x + w].copy()

            if x >= 0 and y >= 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
                hand_crop = img[y:y + h, x:x + w].copy()

                # Convert cropped hand to grayscale
                hand_crop_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)

                equalized = cv2.equalizeHist(hand_crop_gray)

                average_pixel = np.mean(hand_crop_gray)

                hand_crop_gray = reduce_highlights(hand_crop_gray)

                # Compute the scaling factor
                scaling_factor = 128 / average_pixel

                # Apply color correction
                hand_crop_gray = np.clip(hand_crop_gray * scaling_factor, 0, 255).astype(np.uint8)

                hand_crop_gray = cv2.GaussianBlur(hand_crop_gray, (5, 5), 0)
                hand_crop_gray = cv2.bilateralFilter(hand_crop_gray, 9, 75, 75)

                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

                # Apply the sharpening filter
                hand_crop_gray = cv2.filter2D(hand_crop_gray, -1, sharpen_kernel)

                # Convert colors close to black to complete black
                hand_crop_gray[hand_crop_gray < black_threshold] = 0

                # Convert colors close to white to complete white
                hand_crop_gray[hand_crop_gray > white_threshold] = 255

                hand_crop_gray = cv2.bitwise_not(hand_crop_gray)

                predicted_gesture = getPredictedClass(model, hand_crop_gray)

        return render_template('prediction.html', predicted_gesture=predicted_gesture)

    return render_template('predict_image.html')


# Route for getting the predicted gesture
@app.route('/get_prediction')
def get_prediction_route():
    frame = None  # Replace None with the actual frame from the video capture
    gesture = get_prediction(frame)
    return gesture

if __name__ == '__main__':
    app.run(debug=True)