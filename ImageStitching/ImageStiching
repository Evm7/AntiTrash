import os
import cv2
import numpy as np
import imutils
import argparse

from imutils import paths

__author__ = 'EsteveValls'

class ImageStititching():
    def __init__(self, input, output, crop =True, visualize=True, scale =3):
        self.input = input
        self.output = output
        self.crop = crop
        self.visualize=visualize
        self.scale = scale

    def showImage(self, image, option):
        # display the output stitched image to our screen
        h, w, _ = image.shape
        resized_frame = cv2.resize(image, (int(h / self.scale), int(w / self.scale)),interpolation=cv2.INTER_NEAREST)
        cv2.imshow(option, resized_frame)
        cv2.waitKey(0)

    def processDirectory(self):
        # grab the paths to the input images and initialize our images list
        print("[INFO] loading images...")
        imagePaths = sorted(list(paths.list_images(self.input)))
        images = []

        # loop over the image paths, load each one, and add them to our
        # images to stich list
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            images.append(image)

        # initialize OpenCV's image sticher object and then perform the image
        # stitching
        print("[INFO] stitching images...")
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        (status, stitched) = stitcher.stitch(images)

        # if the status is '0', then OpenCV successfully performed image
        # stitching
        if status == 0:
            # check to see if we supposed to crop out the largest rectangular
            # region from the stitched image
            if self.crop:
                # create a 10 pixel border surrounding the stitched image
                print("[INFO] cropping...")
                stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

                # convert the stitched image to grayscale and threshold it
                # such that all pixels greater than zero are set to 255
                # (foreground) while all others remain 0 (background)
                gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)

                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

                # find all external contours in the threshold image then find
                # the *largest* contour which will be the contour/outline of
                # the stitched image
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)

                # allocate memory for the mask which will contain the
                # rectangular bounding box of the stitched image region
                mask = np.zeros(thresh.shape, dtype="uint8")

                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

                # create two copies of the mask: one to serve as our actual
                # minimum rectangular region and another to serve as a counter
                # for how many pixels need to be removed to form the minimum
                # rectangular region
                minRect = mask.copy()
                sub = mask.copy()

                # keep looping until there are no non-zero pixels left in the
                # subtracted image
                while cv2.countNonZero(sub) > 0:
                    # erode the minimum rectangular mask and then subtract
                    # the thresholded image from the minimum rectangular mask
                    # so we can count if there are any non-zero pixels left
                    minRect = cv2.erode(minRect, None)
                    sub = cv2.subtract(minRect, thresh)

                # find contours in the minimum rectangular mask and then
                # extract the bounding box (x, y)-coordinates
                cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(c)

                # use the bounding box coordinates to extract the our final
                # stitched image
                stitched = stitched[y:y + h, x:x + w]

            # write the output stitched image to disk
            cv2.imwrite(self.output +"output.png", stitched)

            # display the output stitched image to our screen
            h, w, _ = stitched.shape
            resized_frame = cv2.resize(stitched, (int(h/self.scale), int(w/self.scale)), interpolation=cv2.INTER_NEAREST)

            if self.visualize:
                cv2.imshow("Stitched", resized_frame)
                cv2.waitKey(0)

            return resized_frame


        # otherwise the stitching failed, likely due to not enough keypoints)
        # being detected
        else:
            print("[INFO] image stitching failed ({})".format(status))

class Counter():
    def __init__(self, image, output, visualize=True):
        self.visualize=visualize
        self.image = image
        self.output = output

    def process_directory(self):
        items, boxes = self.searchSquare(self.image)
        print("[RESULTS] items : "+str(items))
        return items, boxes

    def maskImage(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold of red in HSV space
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1)

        # Threshold of blue in HSV space
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        # preparing the mask to overlay
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        res_red = cv2.bitwise_and(image, image, mask=mask_red)
        res_blue = cv2.bitwise_and(image, image, mask=mask_blue)

        return (res_red, res_blue)

    def prove(self, image):
        # smooth the image with alternative closing and opening
        # with an enlarging kernel
        morph = image.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        return morph


    def detectSquare(self, image, option):
        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        image = self.prove(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(self.output+"Thresh_"+str(option)+".png", thresh)
        cv2.imshow("thresh", thresh)

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        ratio = 1
        boxes = []
        # loop over the contours
        i=0
        items = 0
        for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                # Shortlisting the regions based on there area.
                area = cv2.contourArea(c)
                if area < 400:
                    continue
                items +=1
                #print("Area for "+ str(i) +" is "+ str(area))
                if len(approx) == 4:
                    x1, y1, w, h = cv2.boundingRect(approx)
                    bounding_box = ((x1, y1), (x1 + w, y1 + h))
                    boxes.append(bounding_box)
                    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                    cv2.putText(image, str(i)+ ": "+ str(option)+ ": "+ str(len(boxes)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                else:
                    cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
                    cv2.putText(image,str(i)+ ": Strange " +str(len(approx)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                i+=1


        if self.visualize:
            cv2.imshow('Objects of ' +str(option)+' found', image)
            cv2.imwrite(self.output+"Final"+str(option)+".png", image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return boxes, items

    def searchSquare(self, image):
        print("[INFO] masking panoramic image")
        red, blue = self.maskImage(image)
        boxes ={}
        final_decisor = {}
        print("[INFO] detecting trash")
        boxes["Trash"], final_decisor["Trash"] = self.detectSquare(red, "Trash")
        print("[INFO] detecting valuable")
        boxes["Valuable"], final_decisor["Valuable"] = self.detectSquare(blue, "Valuable")
        return final_decisor, boxes

class CleanlinessLevel():
    def __init__(self, objects, max_objects = 5, threshold = 1):
        self.objects = objects     # dictionray with "valuable" and "trash" keys and the number of items of each.
        self.threshold = threshold
        self.max_obj = max_objects

    def computeLevel(self):
        total_items = self.objects["Trash"] + self.objects["Valuable"]
        level = int(min(1, total_items/self.max_obj) * 10)
        print("[RESULTS] classifying level: "+ str(level))


def main():

    # Import arguments into the python script
    parser = argparse.ArgumentParser("Image Stitching")
    parser.add_argument("-i","--directory", required=True, help="Path to the directory of images to be stitched")
    parser.add_argument('--visualize', type=bool, default=True, help='Introduce True to display the images step by step')
    parser.add_argument('--crop', type=bool, default=False, help='Whether to crop out largest rectangular region')
    parser.add_argument('--output', type=str, default="output", help='Introduce the name of the output final image')
    parser.add_argument('--scale', type=int, default=6, help='Introduce the scale to resize the final image')

    args = parser.parse_args()

    # Parse arguments into variables
    visualize = args.visualize
    input_path = args.directory
    output = args.output
    if str(output).endswith('/') == False:
        output= str(output)+ "/"

    try:
        os.makedirs(output)
        print("[INFO] Creating directory: " + output +" ...")
    except FileExistsError:
        print("[INFO] Directory already exisits ...")

    crop = args.crop
    scale = args.scale

    # Stitching the images
    stitcher = ImageStititching(input_path, output, crop=crop, visualize=visualize , scale=scale)
    image = stitcher.processDirectory()

    # Counting objects
    counter = Counter(image, output,  visualize=visualize)
    objects, _ = counter.process_directory()
    cleanLevel = CleanlinessLevel(objects)
    cleanLevel.computeLevel()

if __name__ == "__main__":
    main()
