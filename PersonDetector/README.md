# Person-Detector
Repository to hold the AntiTrash Project

## Work Done ⏳ :

- [x] CNN Object Detector : YOLOv3 and YOLOv4
- [x] Python - C Wrapper.
- [x] Validation system using HOG Detection Algorithm
- [x] Read Results and Count People
- [x] Classify Images in either Empty of Occupied
- [x] Improve Detection Results by Using YOLO v4
- [x] Get Accuracy of the resukls and apply thresholding technique
- [x] Integrate traffic light color detection
- [x] Enable the detection of whole directory

## Introduction:

One of the main difficulties faced by shared vehicle companies (Carsharing) and public transport is to maintain the vehicle interior in optimal conditions when the user changes, to improve the experience of the journey.

As part of the [AntiTrash project](http://www.carnetbarcelona.com/index.php/2020/04/26/antitrash-a-change-of-model-in-the-maintenance-of-shared-vehicles/), a system will be developed to automatically detect rubbish or potential damage in the shared vehicle or public transport. Through machine learning techniques, images will be used that are obtained by a camera that inspects the interior of the vehicle to assess its state. This will also enable identification of the person responsible for the damage and/or the rubbish within the shared vehicle so that they can be directly charged maintenance costs, if necessary.

For the correct performance of the cleanliness level classifying algorithm, an image pre-processing is necessary to guarantee that, in said photograph, no person can neither be recognized nor appear in the vehicle. In order to confirm such assumption, it is necessary to apply a “person” detector using AI that allows the classification of the image as "empty" or "occupied" vehicles.

## Algorithm:
Such algorithm, which will be described in the next sections, will execute a series of detection methods (YOLO and / or Haar Cascade) of objects to achieve a final result that will indicate whether or not there are passengers inside the shared vehicle.

When creating the experimental prototype that is responsible for the detection of passengers inside the vehicle, it is necessary to specify a series of characteristics that clearly define the final algorithm:
  - The algorithm will be executed in the internal device of the vehicle. A Raspberry Pi 3B is currently being used for project development.
    - The Raspberry model used does not have a GPU, so image processing has a much lower speed. Consumed processing time of the algorithm is limited: maximum time stipulated to analyse an image is 1 minute.
    - Raspberry PI 3b’s memory is very low, limiting the neural networks adaptable to these devices. For the development of the project, a multitude of CNN have been tested: MaskRCNN, Detectron and different versions of YOLO. Due to the large amount of memory and resources required by the program, excessive memory consumption occurred when trying all of them. Therefore, program did not function properly. YOLOv3 and, later on, v4 ,Tiny Version had to be used in order to process images in the Raspberry.
  - The result of the final algorithm is either 0 (there are no passengers in the vehicle in question) or 1 (there are passengers). If the final aim consists on avoiding the transfer of images to the server with recognizable people, it is necessary to differentiate between two types of errors:


## Pre-requisites for its use in Raspberry Pi 3B
1) Python 3.6
3) Numpy `pip3 install numpy`
4) Imutils `pip3 install imutils`
5) OpenCV 3.x with Python bindings. (Tested on OpenCV 3.4.6.27)
    - Newer versions (>3.4.6.27) of Open CV may result in incompatibilities when using the 3B Raspberry Pi version.
6) CUDA/10.0 availability and CUDNN/7.4 (optional --> used to speed up the process)

## Installation:

In order to develop our project we have made use of some existing libraries:
    - [Yolov4 Daknet - Detector] (https://github.com/AlexeyAB/darknet) : Detection algorithm which was created in C language by DarkNet, but I have used an integration command line system  with python due to project languages requirements. Used to detect objects in stipulated frames.
    
Please do install some files in order to execute the code as expected:
    - YOLOv4 Tiny Weights and COCO configurations modules: download from [here](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
    
Create virtual environment and use the requirements.txt for the correct versions installation :
```
pip install -r requirements.txt
```

## Usage:

There is plenty of arguments variations that can modificate the functioning of the video detector:
```
python PersonDetector.py -i <path_input> [-model <model_name> [--resultsFile <name_txt_file>] [--visualize BOOL] [--threshold <ACCURACY>] [--directory BOOL] [--validate BOOL] [--faceDetector >path_to_XML>]
```

When executing the script for the first time, a Results.json file is created where there will be automatically written with details the total results of all detections made in the image. Details of each argument:

    --input' --> 'Path to input image file'
    --resultsFile' --> 'File with all the information of the image detection'
    --model'  --> 'Model to input'
    --visualize'  --> 'Display the image or not'
    --threshold' --> 'Threshold for the accuract of the detection--> default=0.25'
    --directory' --> 'Introduce True if needs recursive detection. Remember to use input with directory'
    --validate' --> 'Introduce True if wants double check through HOG Detector'
    --faceDetector' --> 'Introduce the xml file for the detection (in directory DataFace)'


Example of execution from our servers (it is important to load CUDA/10.0 and CUDNN/7.4 and use the memory consumptions state below):
```
  python PersonDetector.py -i image.jpg --validate True
```  
  
