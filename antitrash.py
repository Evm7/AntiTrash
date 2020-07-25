import subprocess
import sys, os
import json
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


class AntiTrash():
    def __init__(self):
        # Initialize variables
        self.initiate_command = "./darknet"
        self.cfg = "cfg/yolov3-tiny.cfg"
        self.weights = "yolov3-tiny.weights"
        self.arguments = "-ext_output -dont_show"
        self.output_text = "detections.txt"
        self.input_image = "captured_image.jpg"
        self.visualize = False
        self.threshold = 0.25
        self.objects= {}
        self.validate=  False
        self.argument_parsing()


    def argument_parsing(self):
        """
       Parse all the arguments introduced by command line when executing the detection
       :return: arguments initialization
       """

        # HANDLING ARGUMENTS
        import argparse

        parser = argparse.ArgumentParser(description='Process an image and classify human detection.')
        parser.add_argument('-i', '--input', type=str, default="captured_image.jpg", help='Path to input image file')
        parser.add_argument('--resultsFile', type=str, default='detections.txt', help='File with all the information of the image detection')
        parser.add_argument('--model', type=str, default="yolov3-tiny", help='Model to input')
        parser.add_argument('--visualize', type=bool, default=False, help='Display the image or not')
        parser.add_argument('--threshold', type=float, default=0.25, help='Threshold for the accuract of the detection--> default=0.25')
        parser.add_argument('--directory', type=bool, default=False, help='Introduce True if needs recursive detection. Remember to use input with directory')
        parser.add_argument('--validate', type=bool, default=False, help='Introduce True if wants double check through HOG Detector')
        parser.add_argument('--faceDetector', type=str, default="DataFace/haarcascade_profileface.xml", help='Introduce the xml file for the detection (in directory DataFace)')

        args = parser.parse_args()
        self.cfg = "cfg/" + str(args.model) + ".cfg"
        self.weights = str(args.model) + ".weights"
        self.output_text = args.resultsFile
        self.input_image = args.input
        self.visualize = args.visualize
        self.threshold = args.threshold
        self.input_path = args.directory
        self.validate = args.validate
        self.classifier = args.faceDetector

    def execute_command(self):
        self.command =self.initiate_command+" detect " + self.cfg + " " + self.weights + " " + self.input_image + " " + self.arguments
        print("Executing command: " +str(self.command))
        results = subprocess.run(str(self.command).split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return results.stdout.decode('utf-8')


    def read_detections(self, output_text = None):
        """
        Import all the results from the File self.output_text
        """
        if output_text is not None:
            self.output_text = output_text
        try:
            with open(self.output_text) as f:
                return str(f.read())
        except Exception as ex:
            print("There was an error while detecting: " + str(output_text))
            print(str(ex))
        return []

    def transformResults(self, text):
        try:
            start_position = str(text).index(str(self.input_image))
            interesting = str(text)[start_position:-1]
            self.findTime(str(interesting))
            lines = interesting.split("\n")

        except Exception as ex:
            print("Error: Could not parse the results file")
            print(str(ex))
        for line in lines[1:]:
            self.toObject(line)

    def toObject(self, object):
        """
        Format is --> "category" : "accuracy"%\t(left_x:  "left"   top_y:  "top"   width:  "width"   height:  "height")
        """
        results = object.split(" ")
        category = results[0].replace(":","")
        accuracy = results[1].split("%")[0]
        left, top, width, height = self.getBoxes(object)
        id = len(self.objects)
        new_obj = ObjectDetected(id, category, accuracy, left, top, width, height)
        self.objects[id] = new_obj

    def findTime(self, text):
        self.time = self.find_between(text, "Predicted in ", " milli-seconds.")


    def getBoxes(self, text):
        left = self.find_between(text, "left_x:", "top_y").replace(" ", "")
        top = self.find_between(text, "top_y:", "width").replace(" ", "")
        width = self.find_between(text, "width:", "height").replace(" ", "")
        height = self.find_between(text, "height:", ")").replace(" ", "")
        return left, top, width, height


    def find_between(self, s, first, last):
        try:
            start = s.index(first) + len(first)
            end = s.index(last, start)
            return s[start:end]
        except ValueError:
            return ""

    def process_image(self):
        print("Processing image : " + str(self.input_image))
        result = self.execute_command()
        self.transformResults(result)
        self.visualize_image()
        return str(self.objects)

    def visualize_image(self):
        if self.visualize:
            image = cv2.imread("predictions.jpg")

            cv2.imshow("Detected image", image)
            k = cv2.waitKey(0)
            if k == 0xFF:
                pass

    def checkPeople(self):
        ids_ = []
        for id, object in self.objects.items():
            if str(object.category) != 'person':
                ids_.append(id)
            elif float(object.accuracy) < self.threshold:
                ids_.append(id)
        for id in ids_:
            del self.objects[id]

    def classify(self):
        self.checkPeople()
        length_yolo = len(self.objects)
        if self.validate:
            length = self.hog_persondetector(self.input_image)
            print("HOG detector has validated "+ str(length)+" person in the image")
            length_yolo += length
        if length_yolo>0:
            print("We have detected "+ str(len(self.objects))+" person in the image "+ str(self.input_image))
            return 1
        else:
            print("No person has been detected in the image "+ str(self.input_image))
            return 0

    def process_directory(self):
        """
        Main script if making recursive processing of videos for whole directory
        :return:
        """
        import glob
        path = self.input_image
        if not str(self.input_image).endswith("/"):
            path = self.input_image + "/"
        # List all .jpg and .png images in the input directory path and its subdirectories
        files = [f for f in glob.glob(path + "**/*.png", recursive=True)]
        files.extend([f for f in glob.glob(path + "**/*.jpg", recursive=True)])

        num_max = len(files)
        if num_max == 0:
            print("Error: directory introduced did not have any .png or .jpf image in it")
            sys.exit()
        else:
            print("Images we will process:" + str(files).replace('[', '\n\t-').replace(']', '\n').replace(',','\n\t-').replace(" ", ""))
            filename = self.write_file(files)
            if filename is not None:
                self.command = self.initiate_command + " detector test cfg/coco.data " + self.cfg + " " + self.weights + " " + self.arguments + " -out result.json"
                myfile = open(str(filename))
                print("We have started processing the images. Process will last around: " +str(num_max*30)+" seconds")
                results = subprocess.run(str(self.command).split(" "), stdin=myfile, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("All image processing has been executed: now we will classify all images")
                result = self.transformJson()
                self.remove_file(filename)
                return result

    def transformJson(self):
        self.books = (json.loads(self.read_detections("result.json")))
        result = {}
        for image in self.books:
            self.input_image = image["filename"]
            i=0
            self.objects = {}
            for obj in image["objects"]:
                self.objects[i] = ObjectDetected(i,  obj["name"], obj["confidence"], obj["relative_coordinates"]["center_x"], obj["relative_coordinates"]["center_y"], obj["relative_coordinates"]["width"], obj["relative_coordinates"]["height"])
                i+=1
            result[self.input_image] = self.classify()
        print("Final results for all the images :" + str(result).replace('[', '\n\t-').replace(']', '\n').replace(',','\n\t-').replace(" ", ""))
        print("For more information, use the 'result.json' file")
        return result

    def write_file(self, file_list, filename="File.txt"):
        """
        Writes down all files into a .txt --> create the output file that will be read by darknet
        :param directory: name of the directory where document will be placed in
        :param configurations: text to write down
        :return:
        """
        configurations = str(file_list).replace('[', '').replace(']', '').replace(',','\n').replace("'","").replace(" ","")
        try:
            f = open(filename, "x+")
            f.write(configurations)
            return filename
        except:
            print("There was an error while creating file to write configurations of the detection\nError automatically resolved!")
            self.remove_file(filename)
            return self.write_file(file_list, filename)

    def remove_file(self, file_name):
        if os.path.exists(file_name):
            os.remove(file_name)
        else:
            print("The file does not exist")

    def getRecursive(self):
        return self.input_path

    def getAccuracy(self):
        print("accuracy")

    def hog_persondetector(self,filename):
        self.hog = FaceDetectionCV(classifier=self.classifier, visualize=self.visualize)
        # loading image
        image = cv2.imread(filename)
        # resizing for faster detection
        #frame = cv2.resize(image, (640, 480))
        length = self.hog.process_image(image)
        return length

class FaceDetectionCV:
    def __init__(self, classifier="DataFace/haarcascade_frontalface_default.xml", visualize=False):
        print("Starting validation using "+str(classifier)+ "...")
        self.visualize=visualize
        self.haar_cascade_face = cv2.CascadeClassifier(str(classifier))

    def process_image(self, image):
        # call the function to detect faces
        print("Detecting faces...")
        num_faces, faces = self.detect_faces(self.haar_cascade_face, image)
        print("Faces detected: "+str(num_faces))

        if self.visualize:
            # convert to RGB and display image
            cv2.imshow("Face Detector", self.convertToRGB(faces))
            key = cv2.waitKey(0)
            if key == ord('q'):
                pass
        return len(num_faces)

    def convertToRGB(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def detect_faces(self, cascade, test_image, scaleFactor=1.2):
        # convert the test image to gray scale as opencv face detector expects gray images
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        # Applying the haar classifier to detect faces
        faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 15)

        return faces_rect, test_image

class ObjectDetected():
    def __init__(self, id,  category, accuracy, left_x, top_y, width, height):
        # Initialize variables
        self.id = id
        self.category=category
        self.accuracy = accuracy
        self.rect = (int(int(left_x)-int(width)/2), int(int(top_y)+int(height)/2),int(int(left_x)+int(width)/2),  int(int(top_y)-int(height)/2))# (left, bottom, right, top)


    def __str__(self):
        return "ID "+ str(self.id)+ ": "+ str(self.category) + " with accuracy " + str(self.accuracy) + " and position " + str(
            self.rect) + "\n"

    def __repr__(self):
        return "ID "+ str(self.id)+ ": "+ str(self.category) + " with accuracy " + str(self.accuracy) + " and position " + str(
            self.rect) + "\n"


if __name__ == "__main__":
    antitrash = AntiTrash()
    # Check if we are actually asked for recursive Detection or just One vid detection

    if antitrash.getRecursive():
        antitrash.process_directory()
    else:
        print(str(antitrash.process_image()))
        antitrash.classify()

