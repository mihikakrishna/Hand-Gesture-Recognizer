# Hand Gesture Recognition (22)
### Hi Grader 👋 
<b>Fun Fact:</b> We were the <b>only</b> group out of 36 groups to have a live VISUAL demo on presentation day 🤯 and it worked seemlessly well. 

## About The Project
![](https://github.com/mihikakrishna/ECS171-Project/blob/main/demo/live_demo.gif)

Our ECS171 project is a joint effort by a team of five passionate individuals aiming to understand how humans interact with computers. We built a hand gesture recognition system that we implemented on a website. By combining our knowledge of deep learning techniques and computer vision algorithms, our objective was to create a system capable of accurately recognizing and categorizing various hand gestures in real-time scenarios. Through careful testing and continuous refinement, we assessed the performance of our hand gesture recognition system using a diverse range of hand gestures. The outcomes we obtained illustrate the exciting possibilities of improving human-computer interaction through intuitive and effortless control using gestures in different applications. The possibilities are endless.

## Getting Started

### Preparing the ML Model
To set up the ML model for the static and live demos, please follow these steps:
1. Download the CNN model file named `hand_gesture_recognition_6.h5` from [this link](https://drive.google.com/drive/folders/1fmsdW8WjGDg14dz2nC_h0zg_p6d907DZ?usp=sharing).
2. Alternatively, you can run the `train_model.ipnyb` notebook to generate your own `hand_gesture_recognition_6.h5` file.
3. Once you have obtained the `hand_gesture_recognition_6.h5` file either from step 1 or 2, place a copy of it in both the root directory and the `MLModel/` directory.

### Running the Static Demo
To run the static demo, please follow these steps:
1. Navigate to the project directory.
2. Open a terminal or command prompt and enter the following command:
   ```
   python3 app.py
   ```
   This assumes you have Python 3 installed on your system.
3. Once the server is running successfully, you can access the web app at [localhost:5000](http://127.0.0.1:5000)

### Running the Live Demo
To run the live demo, please follow these steps:
1. Navigate to the `MLModel/` directory.
2. Open a terminal or command prompt and enter the following command:
   ```
   python3 main.py
   ```
   This assumes you have Python 3 installed on your system.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 
