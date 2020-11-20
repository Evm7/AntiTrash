# USAGE
# python train.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
from config import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


class Trainer():
    def __init__(self):
        # derive the paths to the training, validation, and testing
        # directories
        self.trainPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])
        self.valPath = os.path.sep.join([config.BASE_PATH, config.VAL])
        self.testPath = os.path.sep.join([config.BASE_PATH, config.TEST])

        # determine the total number of image paths in training, validation,
        # and testing directories
        self.totalTrain = len(list(paths.list_images(self.trainPath)))
        self.totalVal = len(list(paths.list_images(self.valPath)))
        self.totalTest = len(list(paths.list_images(self.testPath)))

        # initialize the training data augmentation object
        self.trainAug = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")

        # initialize the validation/testing data augmentation object (which
        # we'll be adding mean subtraction to)
        self.valAug = ImageDataGenerator()

        # define the ImageNet mean subtraction (in RGB order) and set the
        # the mean subtraction value for each of the data augmentation
        # objects
        self.mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        self.trainAug.mean = self.mean
        self.valAug.mean = self.mean

        # initialize the training generator
        self.trainGen = self.trainAug.flow_from_directory(
            self.trainPath,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=True,
            batch_size=config.BATCH_SIZE)

        # initialize the validation generator
        self.valGen = self.valAug.flow_from_directory(
            self.valPath,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=False,
            batch_size=config.BATCH_SIZE)

        # initialize the testing generator
        self.testGen = self.valAug.flow_from_directory(
            self.testPath,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=False,
            batch_size=config.BATCH_SIZE)

    def plot_training(self, H, N, plotPath):
        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(plotPath)

    def createNetwork(self):
        # load the VGG16 network, ensuring the head FC layer sets are left
        # off
        self.baseModel = VGG16(weights="imagenet", include_top=False,
                               input_tensor=Input(shape=(224, 224, 3)))

        # construct the head of the model that will be placed on top of the
        # the base model
        self.headModel = self.baseModel.output
        self.headModel = Flatten(name="flatten")(self.headModel)
        self.headModel = Dense(512, activation="relu")(self.headModel)
        self.headModel = Dropout(0.5)(self.headModel)
        self.headModel = Dense(len(config.CLASSES), activation="softmax")(self.headModel)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        self.model = Model(inputs=self.baseModel.input, outputs=self.headModel)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in self.baseModel.layers:
            layer.trainable = False

        # compile our model (this needs to be done after our setting our
        # layers to being non-trainable
        print("[INFO] compiling model...")
        opt = SGD(lr=1e-4, momentum=0.9)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,
                           metrics=["accuracy"])

    def train(self):
        # train the head of the network for a few epochs (all other layers
        # are frozen) -- this will allow the new FC layers to start to become
        # initialized with actual "learned" values versus pure random
        print("[INFO] training head...")
        self.H = self.model.fit(
            x=self.trainGen,
            steps_per_epoch=self.totalTrain // config.BATCH_SIZE,
            validation_data=self.valGen,
            validation_steps=self.totalVal // config.BATCH_SIZE,
            epochs=50)

        # reset the testing generator and evaluate the network after
        # fine-tuning just the network head
        print("[INFO] evaluating after fine-tuning network head...")
        self.testGen.reset()
        predIdxs = self.model.predict(x=self.testGen,
                                      steps=(self.totalTest // config.BATCH_SIZE) + 1)
        predIdxs = np.argmax(predIdxs, axis=1)
        print(classification_report(self.testGen.classes, predIdxs,
                                    target_names=self.testGen.class_indices.keys()))
        self.plot_training(self.H, 50, config.WARMUP_PLOT_PATH)

        # reset our data generators
        self.trainGen.reset()
        self.valGen.reset()

        # now that the head FC layers have been trained/initialized, lets
        # unfreeze the final set of CONV layers and make them trainable
        for layer in self.baseModel.layers[15:]:
            layer.trainable = True

        # loop over the layers in the model and show which ones are trainable
        # or not
        for layer in self.baseModel.layers:
            print("{}: {}".format(layer, layer.trainable))

        # for the changes to the model to take affect we need to recompile
        # the model, this time using SGD with a *very* small learning rate
        print("[INFO] re-compiling model...")
        opt = SGD(lr=1e-4, momentum=0.9)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,
                           metrics=["accuracy"])

        # train the model again, this time fine-tuning *both* the final set
        # of CONV layers along with our set of FC layers
        self.H = self.model.fit(
            x=self.trainGen,
            steps_per_epoch=self.totalTrain // config.BATCH_SIZE,
            validation_data=self.valGen,
            validation_steps=self.totalVal // config.BATCH_SIZE,
            epochs=20)

    def evaluate(self):
        # reset the testing generator and then use our trained model to
        # make predictions on the data
        print("[INFO] evaluating after fine-tuning network...")
        self.testGen.reset()
        predIdxs = self.model.predict(x=self.testGen, steps=(self.totalTest // config.BATCH_SIZE) + 1)
        predIdxs = np.argmax(predIdxs, axis=1)
        print(classification_report(self.testGen.classes, predIdxs,
                                    target_names=self.testGen.class_indices.keys()))
        self.plot_training(self.H, 20, config.UNFROZEN_PLOT_PATH)

    def saveModel(self):
        # serialize the model to disk
        print("[INFO] serializing network...")
        self.model.save(config.MODEL_PATH, save_format="h5")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.createNetwork()
    trainer.train()
    trainer.evaluate()
    trainer.saveModel()
