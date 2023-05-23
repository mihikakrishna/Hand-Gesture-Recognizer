from flask import Flask, render_template, Response
import cv2
import time

app = Flask(__name__)

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Set the video capture properties
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Function to perform inference and get the predicted gesture
def get_prediction(frame):
    # Perform inference on the frame using the machine learning model
    # Send the frame to the backend for inference and receive the predicted gesture
    gesture = "Predicted Gesture (Dummy Code)"

    return gesture

# Function to capture video frames
def capture_frames():
    last_capture_time = time.time()

    while True:
        # Read a video frame
        ret, frame = video_capture.read()

        # Process the frame (resize, normalize, etc.) if needed

        # Get the predicted gesture
        gesture = get_prediction(frame)

        # Draw the predicted gesture on the frame if needed

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Yield the frame and predicted gesture for streaming to the frontend
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
               b'Gesture: ' + gesture.encode() + b'\r\n')
        
        # Capture a frame every 1 second
        current_time = time.time()
        if current_time - last_capture_time >= 1.0:
            # Perform additional processing or saving of the captured frame if needed
            last_capture_time = current_time

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for streaming video frames
@app.route('/video_feed')
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for getting the predicted gesture
@app.route('/get_prediction')
def get_prediction_route():
    frame = None  # Replace None with the actual frame from the video capture
    gesture = get_prediction(frame)
    return gesture

if __name__ == '__main__':
    app.run(debug=True)
