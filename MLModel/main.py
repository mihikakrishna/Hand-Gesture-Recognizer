import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model

input_shape = (120, 120)

def getPrediction(model, img):
    image = cv2.resize(img, input_shape)
    gray_image = cv2.resize(image, input_shape)

    gray_image = gray_image.reshape(1, 120, 120, 1)

    # Classes
    class_labels = {
        0: "Blank",
        1: "OK",
        2: "Thumbs Up",
        3: "Thumbs Down",
        4: "Punch",
        5: "High Five",
    }

    # Feed to the NN and get prediction
    prediction_array = model.predict_on_batch(gray_image)
    predicted_class = np.argmax(prediction_array, axis=1)[0]

    if predicted_class in class_labels:
        return class_labels[predicted_class]
    else:
        return "Invalid predicted class"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.2)

    # default crop position
    x = 0
    y = 0

    # Define color thresholds for black and white conversion
    black_threshold = 127
    white_threshold = 128

    predictedClass = ""

    # Load the model
    try:
        model = load_model("hand_gesture_recognition_6.h5")
    except Exception as e:
        print(e)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands = detector.findHands(img, draw=False)

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

                hand_crop_gray = cv2.GaussianBlur(hand_crop_gray, (7, 7), 0)
                hand_crop_gray = cv2.bilateralFilter(hand_crop_gray, 9, 75, 75)

                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

                # Apply the sharpening filter
                hand_crop_gray = cv2.filter2D(hand_crop_gray, -1, sharpen_kernel)

                # Convert colors close to black to complete black
                hand_crop_gray[hand_crop_gray < black_threshold] = 0

                # Convert colors close to white to complete white
                hand_crop_gray[hand_crop_gray > white_threshold] = 255

                hand_crop_gray = cv2.bitwise_not(hand_crop_gray)

                (cnts, _) = cv2.findContours(hand_crop_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(cnts) == 0:
                    continue
                else:
                    # based on contour area, get the maximum contour which is the hand
                    segmented = max(cnts, key=cv2.contourArea)

                    # Create a blank canvas to draw the contour
                    canvas = np.zeros_like(hand_crop_gray, dtype=np.uint8)

                    # Draw the hand contour on the canvas
                    cv2.drawContours(canvas, [segmented], -1, 255, thickness=cv2.FILLED)

                predictedClass = (getPrediction(model, hand_crop_gray))

                cv2.imshow("Hand Contour", hand_crop_gray)

        if hands:
            cv2.putText(img, str(predictedClass), (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "", (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()