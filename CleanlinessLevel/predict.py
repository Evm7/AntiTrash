# USAGE
# python predict.py --image dataset/evaluation/Egg/3_137.jpg 

# import the necessary packages
from tensorflow.keras.models import load_model
from config import config
import numpy as np
import argparse
import imutils
import cv2
import glob

class Predictor():
    def __init__(self):
        self.args = self.parse_arguments()
        # load the trained model from disk
        print("[INFO] loading model...")
        self.model = load_model(config.MODEL_PATH)
        self.tp = 0
        self.fp = 0
        self.classes_results = {}
        for cat in config.CLASSES:
            self.classes_results[cat] = {}  # THEORICAL
            for cat2 in config.CLASSES:
                self.classes_results[cat][cat2] = 0  # PREDICTED

    def parse_arguments(self):
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", type=str, required=True,
                        help="path to our input image")
        args = vars(ap.parse_args())
        return args

    def add_result(self, label, file):
        for cat in config.CLASSES:
            if cat in file:
                name = cat
                break
        self.classes_results[name][label]+=1


    def showResults(self, predictions):
        diction = {}
        for i, classname in enumerate(config.CLASSES):
            diction[classname]=predictions[i]
        return diction

    def processImage(self, file, visualize = False):
        # load the input image and then clone it so we can draw on it later
        image = cv2.imread(file)
        output = image.copy()
        output = imutils.resize(output, width=400)

        # our model was trained on RGB ordered images but OpenCV represents
        # images in BGR order, so swap the channels, and then resize to
        # 224x224 (the input dimensions for VGG16)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # convert the image to a floating point data type and perform mean
        # subtraction
        image = image.astype("float32")
        mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
        image -= mean

        # pass the image through the network to obtain our predictions
        preds = self.model.predict(np.expand_dims(image, axis=0))[0]
        print("For image " + str(file)+ " predictions are : "+ str(self.showResults(preds)), flush=True)
        i = np.argmax(preds)
        label = config.CLASSES[i]
        print("Decided for "+ str(file) +" is "+ label)
        if visualize:
            # draw the prediction on the output image
            text = "{}: {:.2f}%".format(label, preds[i] * 100)
            cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
                    # show the output image
            cv2.imshow("Output", output)
            cv2.waitKey(0)
        return label

    def processDirectory(self):
        for file in glob.glob(self.args["image"]+"*/*.jpg"):
            label = self.processImage(file)
            self.add_result(label, file)
        print(self.classes_results, flush=True)

if __name__ == '__main__':
    predictor = Predictor()
    predictor.processDirectory()