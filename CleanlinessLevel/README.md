# Cleanliness-Level research
Repository to hold the AntiTrash Project

## Work Done ⏳ :

- [] Search for a good already annotated database.
- [] Search for a good object classifier pre trained algorithm
- [] Try the CNN in some images.
- [] Create an algorithm which crops the objects manually (as should do the background subtraction method) and send this image to be detected by the CNN.
- [] Evaluate the process in the database from TRAM and VEHICLE and obtain some images.
- [] If the there are a lot of objects that our database can not classify, search for a new database, and train the CNN.



## Introduction:

One of the main difficulties faced by shared vehicle companies (Carsharing) and public transport is to maintain the vehicle interior in optimal conditions when the user changes, to improve the experience of the journey.

As part of the [AntiTrash project](http://www.carnetbarcelona.com/index.php/2020/04/26/antitrash-a-change-of-model-in-the-maintenance-of-shared-vehicles/), a system will be developed to automatically detect rubbish or potential damage in the shared vehicle or public transport. Through machine learning techniques, images will be used that are obtained by a camera that inspects the interior of the vehicle to assess its state. This will also enable identification of the person responsible for the damage and/or the rubbish within the shared vehicle so that they can be directly charged maintenance costs, if necessary.

Moving to a cleanliness level evaluation instead of classification is indeed of great interest for the system, which should be capable of differentiating between a harmless wrapping paper and more major damage like spills or food. As it is very difficult to obtain images with stains, scratches or spills in the Helisnki Car or in the Public Tram, due to the impossibility of creating them on the seats without causing a permanent mark, there is the need of searching for any other method to detect them.

One proposal could be trying to use the background subtraction method to detect any change to the background typical model created. When there is some area that has been detected as different, now we need to develop some classification object algorithm that would be able to detect the typical objects that can be lost:
  -	Trash: cans, bottles. wrap papers, …
  -	Valuables: mobile phone, laptop, bag , 
