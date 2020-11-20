# Cleanliness-Level research
Repository to hold the AntiTrash Project


## Introduction:

One of the main difficulties faced by shared vehicle companies (Carsharing) and public transport is to maintain the vehicle interior in optimal conditions when the user changes, to improve the experience of the journey.

As part of the [AntiTrash project](http://www.carnetbarcelona.com/index.php/2020/04/26/antitrash-a-change-of-model-in-the-maintenance-of-shared-vehicles/), a system will be developed to automatically detect rubbish or potential damage in the shared vehicle or public transport. Through machine learning techniques, images will be used that are obtained by a camera that inspects the interior of the vehicle to assess its state. This will also enable identification of the person responsible for the damage and/or the rubbish within the shared vehicle so that they can be directly charged maintenance costs, if necessary.

Currently, the AntiTrash project already has a system for detecting the level of dirt in the vehicle, classifying the image as "garbage" or "valuable object" based on the image taken by the camera. In addition, through "Back Substraction" techniques it is possible to obtain the precise location of these anomalous objects inside the vehicle.

<p align="center">
  <img width="700" height="470" src="/CleanlinessLevel/images/annotation.png">
</p>

The project, even so, aims to go further and identify and classify these objects according to their typology (mobiles, wallets, cans, bottles, ...). Consequently, it would be possible to obtain an indicator automatically that extracts a global cleaning level based about the vehicle at that precise moment. This is where the Cleaning Level Classification Algorithm comes in.



## Database
Our database consists of a total of 341 images of valuables and 311 of trash of the entire interior of the vehicle. In each of these images a set of objects can appear included within the super categories of "garbage", "valuable" or “empty”. Through the clipping and annotation algorithm, a total of 595 objects have been extracted, which have been classified into a total of 11 categories:

<p align="center">
  <img width="1000" height="730" src="/CleanlinessLevel/images/database.png">
</p>
For each category we have created a 60-20-20 split in order to create the training, validation and testing set.


<p align="center">
  <img width="460" height="300" src="/CleanlinessLevel/images/split.jpg">
</p>

## Methodology
As the database of the annotated obejcts with each typology was only of 595 objects, we tried to use a popular technique called fine tuning that should adapt a pretrained network (Vgg16 in this case) to our target domain. To do so, we have based our investigation on [PyImageSearch tutorial](https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/) and train using “warm up”:
  1. Remove the fully connected nodes at the end of the network (i.e., where the actual class label predictions are made).
  2. Replace the fully connected nodes with freshly initialized ones (a new set of fully connected layers with random initializations).
  3. Freeze earlier CONV layers earlier in the network (ensuring that any previous robust features learned by the CNN are not destroyed).
  4. Start training, but only train the FC layer heads. This training of the network uses a very small learning rate so the new set of fully connected layers can learn patterns from the previously learned CONV layers earlier in the network — this process is called allowing the FC layers to “warm up”.
  5. Unfreeze all of the CONV layers in the network and perform a second pass of training to readapt the whole model to our data set.

<p align="center">
  <img width="460" height="300" src="/CleanlinessLevel/images/vgg16.jpg">
</p>

## Results
As explained in the methodology, various scenarios were tested to gradually improve our model and adapt it based on the results. Initially, the pre-trained network in another target domain did not correctly classify any of the test objects. Fine Tuning was, in this case, the technique that allowed us to substantially improve our augmentation, although clear overfitting results were achieved.
Finally, through the use of online augmentation and dropout we managed to achieve a 92% accuracy in the test set, very high due to the low number of images we had. In the figure below we can see how the warm-up technique prepared the head layers for the target domain, and finnally fine tuning all the inner layers get the network to learn about our targets:

<p align="center">
  <img width="460" height="300" src="/CleanlinessLevel/images/losses.jpg">
</p>
It should be said that our model was perfectly adapted to those categories of objects with a greater number of images, achieving 100% precision in some of the cases. At the end of the training, the evaluation of the network in our validation split showed very good results:

<p align="center">
  <img width="460" height="300" src="/CleanlinessLevel/images/ValidationResults.jpg">
</p>

However, in order to prove that the results are reliable, there is a need of testing our model to new data which has never seen. Below is the confusion matrix of our final model tested in our dataset:


<p align="center">
  <img width="460" height="300" src="/CleanlinessLevel/images/TestingResults.jpg">
</p>

## Conclusion
We have been able to contextualize our problem, inform ourselves about the different avant-garde techniques currently used to overcome it, and we have put them all to the test in our own database. With all this, the results obtained have been excellent, showing a percentage higher than 90% in all metrics.

Future implementations of this model would have to be retrained with a greater number of images in order to reaffirm the proper functioning of the network. Although the acquisition of the images is not complicated, due to the current situation of the Covid-19 pandemic, it has not been possible to increase the dataset.

As a conclusion, we wanted to be truly honest and even though the results are outstanding, both validation and testing datasets are small and the majoritary number of images are similar. That’s why we can not ensure this great real performance of the model if no more data is provided.

