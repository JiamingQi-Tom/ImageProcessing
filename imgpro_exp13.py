#! /usr/bin/env python3
# import cv2
# import sys
import os
cpath = os.path.dirname(os.path.abspath(__file__))
ccpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
# sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script/image_processing')
# import message_filters
# from imgpro_general import points_sorting
from std_msgs.msg import Int32
# from ClusterAlgorithms import *
from cameraClass_ROS import *
from copy import deepcopy


class ImageProcessing(RealSenseRosSet):
    def __init__(self):
        super(ImageProcessing, self).__init__()

        self.fixedNum = 40

        self.camera_calibration_flag = False

        if self.camera_calibration_flag:
            self.Ts = np.loadtxt(ccpath + '/CameraCalibration/Python/dataset/tf_base_to_camera.txt')
        else:
            self.Ts = np.eye(4)

    def run(self):
        while not rospy.is_shutdown():
            frame1 = deepcopy(self.color_image)

            cv2.imshow('frame1', frame1)

            keyboardVal = cv2.waitKey(1)
            if keyboardVal * 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
            elif keyboardVal == ord('s'):
                pass


if __name__ == '__main__':
    rospy.init_node('imgpro_fabric', anonymous=True)
    ip = ImageProcessing()
    ip.run()
