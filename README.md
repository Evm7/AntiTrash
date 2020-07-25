# AntiTrash
Repository to hold the AntiTrash Project

## Introduction:
One of the main difficulties faced by shared vehicle companies (Carsharing) and public transport is to maintain the vehicle interior in optimal conditions when the user changes, to improve the experience of the journey.

As part of the AntiTrash project, a system will be developed to automatically detect rubbish or potential damage in the shared vehicle or public transport. Through machine learning techniques, images will be used that are obtained by a camera that inspects the interior of the vehicle to assess its state. This will also enable identification of the person responsible for the damage and/or the rubbish within the shared vehicle so that they can be directly charged maintenance costs, if necessary.

In this repository we are facing three approaches that will improve the quality of the AntiTrash Project:

### [Vehicle Occupancies](/PersonDetector)

For the correct performance of the cleanliness level classifying algorithm, an image pre-processing is necessary to guarantee that, in said photograph, no person can neither be recognized nor appear in the vehicle. In order to confirm such assumption, it is necessary to apply a “person” detector using AI that allows the classification of the image as "empty" or "occupied" vehicles.

### [Multiple Image Combination](/ImageStitching)

There are two possibilities when capturing images in the public vehicles:
  
  - Using a **Wide Angle Camera**, which enables to capture the whole vehicle wagon with a single image. However, as the image is taken with 360º lens, the picture may be deformed. Therefore, by applying the Classification and Detecton System to the image the accuracy decreases considerably.
  
  - Using a **Linear Camera**. This method consists on creating a linear camera which takes a set of images to the whole wagon in order to input all the spaces of the wagon with more details. As a consequence, the Classification and Detection System carried on each image performes more accuratly. In order to unify the images and its results to avoid multiple detections and reafirm identifications, we propose to create an algorithm that combines all this results.
  
 ### [Cleanliness Level Research](CleanlinessLevel)
Moving to a cleanliness level evaluation instead of classification is indeed of great interest for the system, which should be capable of differentiating between a harmless wrapping paper and more major damage like spills or food. As it is very difficult to obtain images with stains, scratches or spills in the Helisnki Car or in the Public Tram, due to the impossibility of creating them on the seats without causing a permanent mark, there is the need of searching for any other method to detect them.

One proposal could be trying to use the background subtraction method to detect any change to the background typical model created. When there is some area that has been detected as different, now we need to develop some classification object algorithm that would be able to detect the typical objects that can be lost:
  -	Trash: cans, bottles. wrap papers, …
  -	Valuables: mobile phone, laptop, bag , 

