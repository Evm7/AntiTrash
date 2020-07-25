# Image Stitching
Repository to hold the AntiTrash Project

## Work Done ⏳ :

- [x] Image Stitching
- [x] Mask And Thresholding
- [x] Square Detection

## Introduction:

One of the main difficulties faced by shared vehicle companies (Carsharing) and public transport is to maintain the vehicle interior in optimal conditions when the user changes, to improve the experience of the journey.

As part of the [AntiTrash project](http://www.carnetbarcelona.com/index.php/2020/04/26/antitrash-a-change-of-model-in-the-maintenance-of-shared-vehicles/), a system will be developed to automatically detect rubbish or potential damage in the shared vehicle or public transport. Through machine learning techniques, images will be used that are obtained by a camera that inspects the interior of the vehicle to assess its state. This will also enable identification of the person responsible for the damage and/or the rubbish within the shared vehicle so that they can be directly charged maintenance costs, if necessary.

There are two possibilities when capturing images in the public vehicles:

  - Using a **Wide Angle Camera**, which enables to capture the whole vehicle wagon with a single image. However, as the image is taken with 360º lens, the picture may be deformed. Therefore, by applying the Classification and Detecton System to the image the accuracy decreases considerably.

  - Using a **Linear Camera**. This method consists on creating a linear camera which takes a set of images to the whole wagon in order to input all the spaces of the wagon with more details. As a consequence, the Classification and Detection System carried on each image performes more accuratly. In order to unify the images and its results to avoid multiple detections and reafirm identifications, we propose to create an algorithm that combines all this results: the __Image Stitching__ algorithm
  
  
## Algorithm:
Such algorithm will be taking as input a set of images such as the ones below captured from the linear camera:

![Original Linear Camera Images for the public vehicle](/images/original.png)
Format: ![Alt Text](url)

With this images, an image stitching algorithm will be executed in order to unify all images searching for resemblances between each of them.

![Stitched Images: comparision between cropped or not](/images/imageStitching.png)
Format: ![Alt Text](url)

Then, by applying a thresholding technique and a mask, we will detect the squares and the objects located in the images.

![Final Detections of the Objects in the images](/images/squaresDetected.png)
Format: ![Alt Text](url)


## Pre-requisites for its use in Raspberry Pi 3B
1) Python 3.6
3) Numpy `pip3 install numpy`
4) Imutils `pip3 install imutils`
5) OpenCV `pip3 install opencv-contrib-python`

## Installation:

Create virtual environment and use the requirements.txt for the correct versions installation :
```
pip install -r requirements.txt
```

## Usage:

There is plenty of arguments variations that can modificate the functioning of the video detector:
```
python ImageStitching.py -i <DIRECTORY> [--visualize BOOL] [--output <OUTPUT_NAME>] [--crop BOOL]  [--scale <SCALE>]
```

    parser = argparse.ArgumentParser("Image Stitching")
    parser.add_argument("-i","--directory", required=True, help="Path to the directory of images to be stitched")
    parser.add_argument('--visualize', type=bool, default=True, help='Introduce True to display the images step by step')
    parser.add_argument('--crop', type=bool, default=False, help='Whether to crop out largest rectangular region')
    parser.add_argument('--output', type=str, default="output", help='Introduce the name of the output final image')
    parser.add_argument('--scale', type=int, default=6, help='Introduce the scale to resize the final image')
    

When executing the script for the first time, a Results.json file is created where there will be automatically written with details the total results of all detections made in the image. Details of each argument:

    -i' --> Path to the directory of images to be stitched
    --visualize'  --> Introduce True to display the images step by step
    --crop'  --> Whether to crop out largest rectangular region
    --output' --> Introduce the name of the output final image
    --scale' --> Introduce the scale to resize the final image



```
  python PersonDetector.py -i images
```  
  
