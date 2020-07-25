# AntiTrash
Repository to hold the AntiTrash Project

## Introduction:
One of the main difficulties faced by shared vehicle companies (Carsharing) and public transport is to maintain the vehicle interior in optimal conditions when the user changes, to improve the experience of the journey.

As part of the AntiTrash project, a system will be developed to automatically detect rubbish or potential damage in the shared vehicle or public transport. Through machine learning techniques, images will be used that are obtained by a camera that inspects the interior of the vehicle to assess its state. This will also enable identification of the person responsible for the damage and/or the rubbish within the shared vehicle so that they can be directly charged maintenance costs, if necessary.

For the correct performance of the cleanliness level classifying algorithm, an image pre-processing is necessary to guarantee that, in said photograph, no person can neither be recognized nor appear in the vehicle. In order to confirm such assumption, it is necessary to apply a “person” detector using AI that allows the classification of the image as "empty" or "occupied" vehicles.

Algorithm:
