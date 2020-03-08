# Project Title: OpenVino Emotion Analysis

This project aims at building a pipeline that identifies each student/lecturer in the classroom and captures the emotions of each one of them. In the end, it generates the report card of each student/lecturer which will tell the overall fluctuations in the behaviour of the student/lecturer through the day.


## Prerequisites

1. Download and install [OpenVino](https://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_windows.html) along with other dependenciy software and run the demo mentioned in the installatin guide to check everything installed properly.[Note: I used openvino version 2019_R3.1]
2. Download and install [Anaconda](https://www.anaconda.com/distribution/#windows) with python 3.x.

## Getting Started

1. Clore this repo
```git
  git clone https://github.com/apthagowda97/openvino_emotion_analysis.git
```
2. Open Command Prompt and change the directory to the cloned repositery
```cmd
  C:\>cd [path]\openvino_emotion_analysis
  C:\[path]\openvino_emotion_analysis>run.bat
  C:\[path]\openvino_emotion_analysis>jupyter notebook
```
# Description
## Phase 1: 
![flowchart](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/flowchart.png)

### 1. Breaks the video data into frames.
![video frames](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/download.png)

### 2. Takes singel frame at a time.
![frame1](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/download%20(1).png)

### 3. Runs the [face-detection-retail-0005](http://docs.openvinotoolkit.org/latest/_models_intel_face_detection_retail_0005_description_face_detection_retail_0005.html)
![frame2](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/download%20(2).png)
![faces](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/download%20(3).png)

### 4. Runs the [age-gender-recognition-retail-0013](http://docs.openvinotoolkit.org/latest/_models_intel_age_gender_recognition_retail_0013_description_age_gender_recognition_retail_0013.html)
![faces_with gender](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/download%20(4).png)

### 5. Runs the [emotions-recognition-retail-0003](http://docs.openvinotoolkit.org/latest/_models_intel_emotions_recognition_retail_0003_description_emotions_recognition_retail_0003.html)
![frame3](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/download%20(5).png)

### 6. Plots the overall emotion of the frame and add that to the frame
![emotion graph](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/download%20(6).png)
![frame](https://github.com/apthagowda97/openvino_emotion_analysis/blob/master/docs/download%20(7).png)





## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
