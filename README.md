<h1 align="center">Volume control project with hand recognition 🎚️</h1>
<h1 align="left">🗂Project introduction</h1>

This project is an intelligent system for recognizing hand gestures and controlling sound in Windows, which uses:
* MediaPipe to detect hand key points
* OpenCV for image processing
* pycaw for Windows sound control

<h1 align="left">🔑Key Features</h1>

* Instant hand detection in the image
* Finger position recognition
* Voice control with hand gestures
* Visual volume display
* High-speed operation

### 📁 File Structure

```
Volume control project with hand recognition
├── HTM.py   
├── VHC.py    
└── README.md
```

### 🛠️Installing the required libraries

```
pip install opencv-python mediapipe numpy comtypes pycaw

```


<h1 align="left">🧑🏻‍💻How to run</h1>

### 1. Hand Recognition Module
* to run the hand recognition demo:
```
python HTM.py
```
### 2. Voice Control Program
* to run the Voice Control Program:
```
python VHC.py
```
<h1 align="left">⚙️Volume control method</h1>

### 1. Place your hand in front of the webcam.
### 2. Adjust the volume by changing the distance between your thumb and forefinger:
* 👌 Small distance: mute
* 🖐️ Large distance: maximum volume
###  3. Press the 'q' key to exit.


## 📜License

[MIT](https://github.com/justbehrad/ControlVolumeByHand/blob/main/LICENSE)
