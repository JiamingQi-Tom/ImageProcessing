#! /usr/bin/env python3
import numpy as np
import cv2
from cameraClass_ROS import *
from time import time
# from method import *
from copy import deepcopy
from General_method1 import find_area


class ImageProcessing(RealSenseRosSet):
    def __init__(self):
        super(ImageProcessing, self).__init__()
        self.publish_rate = 30.0
        self.rate = rospy.Rate(self.publish_rate)

        cv2.namedWindow('Selection')
        cv2.createTrackbar('h', 'Selection', 48, 180, self.nothing)
        cv2.createTrackbar('s', 'Selection', 73, 255, self.nothing)
        cv2.createTrackbar('v', 'Selection', 52, 255, self.nothing)

    def nothing(self, x):
        pass

    def run(self):
        while not rospy.is_shutdown():
            frame1 = deepcopy(self.color_image)

            h = cv2.getTrackbarPos('h', 'Selection')
            s = cv2.getTrackbarPos('s', 'Selection')
            v = cv2.getTrackbarPos('v', 'Selection')

            green, closing = find_area(frame1, np.array([h, s, v]))
            for i in range(np.size(green, 0)):
                cv2.circle(frame1, (green[i, 1], green[i, 0]), 5, (0, 0, 255), -1)

            keyboardVal = cv2.waitKey(1)
            if keyboardVal * 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
            elif keyboardVal == ord('s'):
                pass

            cv2.imshow('frame1', frame1)
            cv2.imshow('Selection', closing)
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('imgpro_fabric', anonymous=True)
    ip = ImageProcessing()
    ip.run()








