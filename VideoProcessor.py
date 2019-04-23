import cv2
import copy
import argparse
import os
import numpy as np
import math
from PIL import Image
import time
import imutils
from AudioGenerator import AudioGenerator
import datetime
import pyaudio

GREEN_ONLY = True

try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    PI_CAMERA = True
except:
    PI_CAMERA = False

class VideoProcessor:
    """
    To run me on the pi in production mode: python3 VideoProcessor.py --greedy --audio
    To run me on the pi w/ video feed: python3 VideoProcessor.py --greedy --audio --render --circles

    To run on a local machine, just include the --dev flag

    The greedy mechanism discards every circle aside from the one closest to the
    center of the image. While not great in design, it seems to work well
    enough for our purposes.
    
############################### DEV CONFIGURATION ##############################
    :capture -> cv2.VideoCapture: video capture object


############################### PI CONFIGURATION ###############################
    :camera -> PiCamera: Pi Camera Module capture object
    :raw_capture -> PiRGBArray: processed capture images
    """
    def __init__(self, boundaries, audiogenerator=None):
        """
        Initialize our class, configure CLI arguments, make sure that 
            there isn't an environment collision
        :boundaries -> [int, int]: x,y dimensions
        :audiogenerator -> AudioGenerator: our audio generator implementation
        """
        global PI_CAMERA
        self.args = self.parse_arguments()
        self.boundaries = boundaries
        self.audio_generator = audiogenerator
        if self.args["dev"] is PI_CAMERA:
            print("It looks like you are trying to run conflicting environments")
            print("If you are on your test machine, add the --dev flag")
            print("If you are on the pi, remove the --dev flag")
            print("If it still fails on the pi, ensure the picamera packages are configured")
            exit(1)
        if self.args["dev"]:
            self.configure_webcam()
        else:
            self.configure_picam()

##################################################################################
###################################### SETUP #####################################
##################################################################################
    
    def configure_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, self.boundaries[0])
        self.capture.set(4, self.boundaries[1])
    
    def configure_picam(self):
        self.camera = PiCamera()
        self.camera.resolution = (self.boundaries[0], self.boundaries[1])
        self.camera.framerate = 32
        self.raw_capture = PiRGBArray(self.camera, size=(self.boundaries[0], self.boundaries[1]))
        time.sleep(0.1)
    
    def parse_arguments(self):
        """
        Configure our command line arguments
        Options are:
            greedy  - discard all found circles aside from one closest to center
            dev     - for local testing, use webcam instead of PiCameraModule
            save    - save all frames
            render  - render the computer vision on screen
            orig    - don't do any processing, for use w/ saving to get raw capture
            circles - only draw circles on image if true
            audio   - generate audio feedback
            help    - show these arguments
        :returns -> {}: dict of flags to their values
        """
        ap = argparse.ArgumentParser()
        ap.add_argument("-g", "--greedy", action="store_true", help="keep only center-most circle")
        ap.add_argument("-d", "--dev", action="store_true", help="use when testing on your local machine")
        ap.add_argument("-s", "--save", action="store_true", help="save all seen frames to cwd/logs")
        ap.add_argument("-r", "--render", action="store_true", help="show output window")
        ap.add_argument("-o", "--original", action="store_true", help="don't do any image processing")
        ap.add_argument("-c", "--circles", action="store_true", help="draw the circles on render")
        ap.add_argument("-a", "--audio", action="store_true", help="generate audio feedback")
        return vars(ap.parse_args())

##################################################################################
################################ IMAGE OPERATIONS ################################
##################################################################################

    def _process_find_circles(self, frame):
        """
        Receive a processed images, and try to identify circles in it.
        If we are using the greedy flag, only return the circle closest
        to the center of the image.
        :frame -> image object: array of the image
        :returns -> frame, [x,y]: where [x,y] is location of circle, or [0,0] if no cirlce
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2.0, 
              minDist=120,
              param1=20,#80,
              param2=40,#80,
              minRadius=0,
              maxRadius=40)
        if circles is not None:
            if self.args["greedy"]:
                bestCircle = self.discard_worst(circles)
                if self.args["circles"]:
                    cv2.circle(frame,(bestCircle[0],bestCircle[1]),bestCircle[2],(0,255,0),2)
                    cv2.circle(frame,(bestCircle[0],bestCircle[1]),2,(0,0,255),3)
                return frame, bestCircle
            else:
                circles = np.uint16(np.around(circles))
                if self.args["circles"]:
                    for i in circles[0,:]:
                        # draw the outer circle
                        cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                        # draw the center of the circle
                        cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                return frame, circles
        else:
            return frame, [0, 0]
    
    def _process_filter_colors(self, frame):
        """
        Receive an image (frame) and apply our colormatch. Any colors
        in the masks will stay in the image, every other pixel will be discarded
        Currently we only operate on bright green, set by the global var.
        :frame -> image object: array of the image
        :returns -> frame: frame w/ only selected colors intact
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        global GREEN_ONLY

        if GREEN_ONLY:
            lower_green = np.array([40, 100, 105])
            upper_green = np.array([90, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            green = cv2.bitwise_and(frame, frame, mask=green_mask)
            return green
        
        else:
            # RED
            lower_red = np.array([0, 140, 120])
            upper_red = np.array([15, 255, 255])
            lower_red_ = np.array([170, 140, 120])
            upper_red_ = np.array([179, 255, 255])
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            red_mask_ = cv2.inRange(hsv, lower_red_, upper_red_)
            
            # BLUE
            lower_blue = np.array([90, 40, 50])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # YELLOW
            lower_yellow = np.array([18, 80, 80])
            # upper_yellow = np.array([35, 255, 255])
            upper_yellow = np.array([35, 240, 200])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            #GREEN
            lower_green = np.array([50, 40, 50])
            upper_green = np.array([80, 180, 180])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            

            red = cv2.bitwise_or(cv2.bitwise_and(frame,frame, mask=red_mask), cv2.bitwise_and(frame, frame, mask=red_mask_))
            blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
            yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)
            green = cv2.bitwise_and(frame, frame, mask=green_mask)
            # return cv2.bitwise_or(red, blue)
            acc = cv2.bitwise_or(red, blue)
            acc = cv2.bitwise_or(acc, green)
            # acc = cv2.bitwise_or(cv2.bitwise_not(black), acc)
            return cv2.bitwise_or(acc, yellow)
    
    def _proc_blur(self, frame):
        """
        Blur the image a little
        :frame -> image object: input frame to blur
        :returns -> frame: frame w/ blur
        """
        return cv2.GaussianBlur(frame,(5,5), 0)
    
    def process_chain(self, frame):
        """
        Filter colors, then blur, then search for circles
        :frame -> image object: array of the image
        :returns -> frame, [x,y]: where [x,y] is location of circle, or [0,0] if no cirlce
        """
        frame = self._proc_blur(self._process_filter_colors(frame))
        return self._process_find_circles(frame)

##################################################################################
################################# RUNNERS/HELPERS ################################
##################################################################################

    def discard_worst(self, circles):
        """
        Get an array of circles, and using a shortest-distance heuristic,
        return the circle w/ the shortest distance to the center of the image
        :circles -> [[x,y,r(adius)], ...]: array of x, y , radius of all circles
        :returns -> [x,y]: coordinates of best circle
        """
        circles = circles[0]
        bestHeur = 9999999999
        bestCircle = None
        for circle in circles:
            curHeur = abs(circle[0] - (self.boundaries[0]/2)) + abs(circle[1] - (self.boundaries[1]/2))
            if curHeur < bestHeur:
                bestCircle = circle
                bestHeur = curHeur
        return bestCircle

    def run(self):
        """
        Detect our environment and run accordingly
        """
        if self.args["dev"]:
            self.run_local()
        else:
            self.run_pi()
    
    def run_local(self):
        # Master run loop for when testing on local machine
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            frame = cv2.resize(frame, (self.boundaries[0], self.boundaries[1]))
            if ret:
                if not self.args["original"]:
                    frame, circles = self.process_chain(frame)
                if self.args["audio"] and self.args["greedy"]:
                    au_target = [0, 0]
                    au_target[0] = int(round(circles[0]))
                    au_target[1] = int(round(circles[1]))
                    self.audio_generator.run(au_target)
                if self.args["save"]:
                    self.save_frame(frame)
                if self.args["render"]:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
    
    def run_pi(self):
        # Master run loop for when running in "production" mode
        for frame in self.camera.capture_continuous(self.raw_capture, format="bgr", use_video_port=True):
            image = frame.array
            image = imutils.rotate(image, 270)

            if not self.args["original"]:
                image, circles = self.process_chain(image)
            if self.args["audio"] and self.args["greedy"]:
                    au_target = [0, 0]
                    au_target[0] = int(round(circles[0]))
                    au_target[1] = int(round(circles[1]))
                    self.audio_generator.run(au_target)
            if self.args["save"]:
                self.save_frame(image)
            if self.args["render"]:
                try:
                    cv2.imshow("Frame", image)
                except:
                    pass
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            self.raw_capture.truncate(0)
    
    def save_frame(self, frame):
        # If using the correct flag, will save an img every second for debugging.
        timeString = str(datetime.datetime.now()).replace(" ", "_")[0:19]
        pathStr = str(os.path.abspath(os.getcwd())) + '/logs/' \
            + timeString + '.png'
        if not os.path.isfile(pathStr):
            cv2.imwrite(pathStr, frame)

if __name__ == "__main__":
    go = VideoProcessor([640, 480], AudioGenerator([640, 480]))
    go.run()