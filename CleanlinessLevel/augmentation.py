import glob
import os
import shutil

import imgaug as ia
from imgaug import augmenters as iaa
from imutils import paths
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys


class Augmentation:
    def __init__(self):
        ia.seed(1)
        self.directory = "Database/"
        txtfiles = []
        for file in glob.glob(self.directory + "*/*.jpg"):
            txtfiles.append(file)
        self.imagePaths = sorted(txtfiles)

    def define_agumentation(self):
        # http://imgaug.readthedocs.io/en/latest/source/augmenters.html#sequential
        self.seq_of_aug = iaa.Sequential([
            iaa.Crop(percent=(0, 0.1)),  # random crops

            # horizontally flip 50% of the images
            iaa.Fliplr(0.5),

            # Gaussian blur to 50% of the images
            # with random sigma between 0 and 0.5.
            iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.5))),

            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),

            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            # Apply affine transformations to each image.
            # Scale/zoom them from 90% 5o 110%
            # Translate/move them, rotate them
            # Shear them slightly -2 to 2 degrees.
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-5, 5),
                shear=(-2, 2)
            )
        ], random_order=True)  # apply augmenters in random order

    def getName(self, path):
        name = path.split("\\")[2]
        return (name.split(".")[0]).split("_")

    def augmentate(self, path="Database/backpack"):
        self.define_agumentation()
        imagePaths = sorted(list(paths.list_images(path)))
        j=0
        # Some example augmentations using the seq defined above.
        for imagePath in imagePaths:

                image = cv2.imread(imagePath)
                classname, id = self.getName(imagePath)
                for i in range(4):
                    image_aug = self.seq_of_aug.augment_image(image)
                    #print(directory+classname+"/"+classname+"_"+str(id)+"_aug"+str(i)+".jpg")
                    cv2.imwrite(self.directory+classname+"/"+classname+"_"+str(id)+"_aug"+str(i)+".jpg", image_aug)




    def getClass(self, path):
        name = path.split("\\")[2]
        return name.split("_")[0], name


    def split_directory(self, previous_directory ="Database", new_directory ="Augm_DataBase"):
    # In order to split the dataset in training, validation and evaluation
        if os.path.isdir(new_directory) is False:
            os.mkdir(new_directory)
        if os.path.isdir(new_directory + "/evaluation") is False:
            os.mkdir(new_directory + "/evaluation")
        if os.path.isdir(new_directory + "/training") is False:
            os.mkdir(new_directory + "/training")
        if os.path.isdir(new_directory + "/validation") is False:
            os.mkdir(new_directory + "/validation")

        imagePaths = sorted(list(paths.list_images(previous_directory)))
        random.shuffle(imagePaths)
        for imagePath in imagePaths:
            classname, name = self.getClass(imagePath)
            num = np.random.rand(1)
            if  num < 0.2: #EVALUATION:
                category = "evaluation"
            elif num<0.4: # VALIDATION
                category = "validation"
            else:
                category = "training"

            if os.path.isdir(new_directory + '/'+category+'/'+classname) is False:
                os.mkdir(new_directory + '/'+category+'/'+classname)
            shutil.copy2(imagePath, new_directory + '/'+category+'/'+classname+"/"+name)

    def delete_augmentation(self, directory):
        imagePaths = sorted(list(paths.list_images(directory)))
        for path in imagePaths:
            if "_aug" in path:
                print("Removing file : "+ str(path))
                os.remove(path)


    def confusion_matrix(self, dict):
        keys = list(dict.keys())
        predicted = {}
        predicted_values = []
        expected_values = []
        for index, k in enumerate(keys):
            # Creation of the ground truth array
            for i in range(0, self.countValues(dict[k])):
                expected_values.append(index)
            # Creation of the predicted vector
            for key, value in dict[k].items():
                if key not in predicted:
                    predicted[key]=0
                predicted[key]+=value
        for index, k in enumerate(keys):
            for i in range(0,int(predicted[k])):
                predicted_values.append(index)
        return predicted_values, expected_values

    def countValues(self, dict2):
        sum = 0
        for k, v in dict2.items():
            sum+=v
        return sum

    def statistics(self, actual, predicted, normalize=False):
        # Not Required, But Nice For Legibility
            """
            Generate a confusion matrix for multiple classification
            @params:
                actual      - a list of integers or strings for known classes
                predicted   - a list of integers or strings for predicted classes
            @return:
                matrix      - a 2-dimensional list of pairwise counts
                statistics  - a dictionary of statistics for each class
            """
            unique = sorted(set(actual))
            matrix = [[0 for _ in unique] for _ in unique]
            imap = {key: i for i, key in enumerate(unique)}
            # Generate Confusion Matrix
            for p, a in zip(predicted, actual):
                matrix[imap[p]][imap[a]] += 1
            # Get Confusion Matrix Sum
            sigma = sum([sum(matrix[imap[i]]) for i in unique])
            # Scaffold Statistics Data Structure
            statistics = OrderedDict(((i, {"counts": OrderedDict(), "stats": OrderedDict()}) for i in unique))
            # Iterate Through Classes & Compute Statistics
            for i in unique:
                loc = matrix[imap[i]][imap[i]]
                row = sum(matrix[imap[i]][:])
                col = sum([row[imap[i]] for row in matrix])
                # Get TP/TN/FP/FN
                tp = loc
                fp = row - loc
                fn = col - loc
                tn = sigma - row - col + loc
                # Populate Counts Dictionary
                statistics[i]["counts"]["tp"] = tp
                statistics[i]["counts"]["fp"] = fp
                statistics[i]["counts"]["tn"] = tn
                statistics[i]["counts"]["fn"] = fn
                statistics[i]["counts"]["pos"] = tp + fn
                statistics[i]["counts"]["neg"] = tn + fp
                statistics[i]["counts"]["n"] = tp + tn + fp + fn
                # Populate Statistics Dictionary
                statistics[i]["stats"]["sensitivity"] = tp / (tp + fn) if tp > 0 else 0.0
                statistics[i]["stats"]["specificity"] = tn / (tn + fp) if tn > 0 else 0.0
                statistics[i]["stats"]["precision"] = tp / (tp + fp) if tp > 0 else 0.0
                statistics[i]["stats"]["recall"] = tp / (tp + fn) if tp > 0 else 0.0
                statistics[i]["stats"]["tpr"] = tp / (tp + fn) if tp > 0 else 0.0
                statistics[i]["stats"]["tnr"] = tn / (tn + fp) if tn > 0 else 0.0
                statistics[i]["stats"]["fpr"] = fp / (fp + tn) if fp > 0 else 0.0
                statistics[i]["stats"]["fnr"] = fn / (fn + tp) if fn > 0 else 0.0
                statistics[i]["stats"]["accuracy"] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn) > 0 else 0.0
                statistics[i]["stats"]["f1score"] = (2 * tp) / ((2 * tp) + (fp + fn)) if tp > 0 else 0.0
                statistics[i]["stats"]["fdr"] = fp / (fp + tp) if fp > 0 else 0.0
                statistics[i]["stats"]["for"] = fn / (fn + tn) if fn > 0 else 0.0
                statistics[i]["stats"]["ppv"] = tp / (tp + fp) if tp > 0 else 0.0
                statistics[i]["stats"]["npv"] = tn / (tn + fn) if tn > 0 else 0.0

            mean_statistics = {}
            def add(dict1, dict2, key):
                if key in dict2["counts"]:
                    value = dict2["counts"][key]
                else:
                    value = dict2["stats"][key]
                if not key in dict1:
                    dict1[key]=0
                dict1[key]+=value

            for k, v in statistics.items():
                add(mean_statistics, v, "tp")
                add(mean_statistics, v, "fp")
                add(mean_statistics, v,  "tn")
                add(mean_statistics, v,  "fn")
                add(mean_statistics, v,  "pos")
                add(mean_statistics, v,  "neg")
                add(mean_statistics, v,  "n")
                add(mean_statistics, v,  "neg")

            tp = mean_statistics["tp"]
            tn = mean_statistics["tn"]
            fp = mean_statistics["fp"]
            fn = mean_statistics["fn"]

            mean_statistics["sensitivity"] = tp / (tp + fn) if tp > 0 else 0.0
            mean_statistics["specificity"] = tn / (tn + fp) if tn > 0 else 0.0
            mean_statistics["precision"] = tp / (tp + fp) if tp > 0 else 0.0
            mean_statistics["recall"] = tp / (tp + fn) if tp > 0 else 0.0
            mean_statistics["tpr"] = tp / (tp + fn) if tp > 0 else 0.0
            mean_statistics["tnr"] = tn / (tn + fp) if tn > 0 else 0.0
            mean_statistics["fpr"] = fp / (fp + tn) if fp > 0 else 0.0
            mean_statistics["fnr"] = fn / (fn + tp) if fn > 0 else 0.0
            mean_statistics["accuracy"] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn) > 0 else 0.0
            mean_statistics["f1score"] = (2 * tp) / ((2 * tp) + (fp + fn)) if tp > 0 else 0.0
            mean_statistics["fdr"] = fp / (fp + tp) if fp > 0 else 0.0
            mean_statistics["for"] = fn / (fn + tn) if fn > 0 else 0.0
            mean_statistics["ppv"] = tp / (tp + fp) if tp > 0 else 0.0
            mean_statistics["npv"] = tn / (tn + fn) if tn > 0 else 0.0

            # Matrix Normalization
            if normalize:
                matrix = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
            return matrix, mean_statistics

    def plot_confusion(self):
        dict ={'backpack': {'backpack': 10, 'can': 0, 'empty': 0, 'headphones': 0, 'keys': 0, 'laptop': 0, 'phone': 0, 'plasticBottle': 0, 'tissue': 0, 'sunglasses': 0, 'wallet': 1}, 'can': {'backpack': 0, 'can': 21, 'empty': 0, 'headphones': 0, 'keys': 0, 'laptop': 0, 'phone': 0, 'plasticBottle': 1, 'tissue': 0, 'sunglasses': 0, 'wallet': 0}, 'empty': {'backpack': 0, 'can': 1, 'empty': 15, 'headphones': 0, 'keys': 0, 'laptop': 0, 'phone': 0, 'plasticBottle': 1, 'tissue': 0, 'sunglasses': 0, 'wallet': 0}, 'headphones': {'backpack': 0, 'can': 0, 'empty': 0, 'headphones': 1, 'keys': 0, 'laptop': 0, 'phone': 0, 'plasticBottle': 0, 'tissue': 0, 'sunglasses': 0, 'wallet': 1}, 'keys': {'backpack': 0, 'can': 1, 'empty': 0, 'headphones': 0, 'keys': 6, 'laptop': 0, 'phone': 0, 'plasticBottle': 0, 'tissue': 0, 'sunglasses': 0, 'wallet': 0}, 'laptop': {'backpack': 0, 'can': 0, 'empty': 0, 'headphones': 0, 'keys': 0, 'laptop': 1, 'phone': 0, 'plasticBottle': 0, 'tissue': 0, 'sunglasses': 0, 'wallet': 0}, 'phone': {'backpack': 0, 'can': 0, 'empty': 0, 'headphones': 0, 'keys': 0, 'laptop': 0, 'phone': 15, 'plasticBottle': 0, 'tissue': 0, 'sunglasses': 0, 'wallet': 0}, 'plasticBottle': {'backpack': 0, 'can': 3, 'empty': 0, 'headphones': 0, 'keys': 1, 'laptop': 0, 'phone': 0, 'plasticBottle': 20, 'tissue': 0, 'sunglasses': 0, 'wallet': 1}, 'tissue': {'backpack': 0, 'can': 1, 'empty': 0, 'headphones': 0, 'keys': 0, 'laptop': 0, 'phone': 0, 'plasticBottle': 0, 'tissue': 0, 'sunglasses': 1, 'wallet': 0}, 'sunglasses': {'backpack': 0, 'can': 0, 'empty': 0, 'headphones': 0, 'keys': 0, 'laptop': 0, 'phone': 0, 'plasticBottle': 0, 'tissue': 1, 'sunglasses': 0, 'wallet': 0}, 'wallet': {'backpack': 0, 'can': 1, 'empty': 0, 'headphones': 0, 'keys': 0, 'laptop': 0, 'phone': 1, 'plasticBottle': 0, 'tissue': 0, 'sunglasses': 0, 'wallet': 7}}
        pred, exp = self.confusion_matrix(dict)
        keys = list(dict.keys())
        import pandas as pd
        y_actu = pd.Series(exp, name='Actual')
        y_pred = pd.Series(pred, name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
        self.plot_confusion_matrix(df_confusion, keys)
        matrix, statistics = self.statistics(y_actu,y_pred, normalize=True)
        import json
        with open('result.json', 'w') as fp:
            json.dump(statistics, fp,indent=4)




    def plot_confusion_matrix(self ,df_confusion, keys, title='Confusion matrix', cmap=plt.cm.gray_r):
        df_confusion=df_confusion / df_confusion.sum(axis=1)
        plt.matshow(df_confusion, cmap=cmap)  # imshow
        # plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(df_confusion.columns))
        plt.xticks(tick_marks, keys, rotation=45)
        plt.yticks(tick_marks, keys)
        # plt.tight_layout()
        plt.ylabel(df_confusion.index.name)
        plt.xlabel(df_confusion.columns.name)
        plt.show()

    def count(self, directory="SeparatedDatabase"):
        imagePaths = sorted(list(paths.list_images(directory)))
        print(len(imagePaths))


if __name__ == '__main__':
    augmentation = Augmentation()
    augmentation.augmentate()
    #augmentation.split_directory("OriginalDatabase","SeparatedDatabase")
    #augmentation.plot_confusion()
    #augmentation.count()
