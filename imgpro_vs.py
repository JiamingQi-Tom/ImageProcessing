
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/tom/Documents/exps_ws/src/plugins/script/ImageProcessing')
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from copy import deepcopy
from ConvertPointcloud import *
from plugins.msg import msg_float, msg_int
from cameraClass_ROS import *


class ImageProcessing(RealSenseRosSet):
    def __init__(self):
        super(ImageProcessing, self).__init__()

        self.feature_2D = []
        self.pub1 = rospy.Publisher("/feature_2D", msg_int, queue_size=2)
        
        self.feature_3D = []
        self.pub2 = rospy.Publisher("/feature_3D", PointCloud2, queue_size=2)

        # ---------------------------------- Configuration for aruco Marker ---------------------------------- #
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        rospy.sleep(1)

        cv2.namedWindow("frame1")

    def run(self):
        while not rospy.is_shutdown():
            frame1 = deepcopy(self.color_image)

            # Read intrinsic parameters of the RGB camera, not depth camera #
            intr_matrix = self.color_intrinsics
            intr_coeffs = np.array(self.color_camera_info.D)

            try:

                # Check how many ArUco markers detected #
                markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(frame1)
                if len(markerCorners) > 0:

                    for i in range(len(markerCorners)):
                        # Step 1: Get pose for each detected marker; the second parameter is marker length #
                        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(markerCorners[i], 0.09, intr_matrix, intr_coeffs)

                        # Step 2: Draw a rectangle for the marker #
                        aruco.drawDetectedMarkers(frame1, markerCorners)

                        # Step 3: Draw the axis for the marker #
                        cv2.drawFrameAxes(frame1, intr_matrix, intr_coeffs, rvec, tvec, 0.09)

                        # Step 4: Fill in rotation matrix #
                        # rotation_matrix, _ = cv2.Rodrigues(rvec)
                        # self.tf_camera_to_marker[:3, :3] = rotation_matrix

                        # Step 5: Fill in translation vector #
                        # self.tf_camera_to_marker[:3, 3] = tvec.squeeze()

                        # Step 6: Calculated four corners of each ArUco marker #
                        (topLeft, topRight, bottomRight, bottomLeft) = markerCorners[i].reshape((4, 2)).astype(int)

                        self.feature_2D = deepcopy(markerCorners[i].reshape((4, 2)).astype(int))
                        for pIdx in range(4):
                            cv2.circle(frame1, (self.feature_2D[pIdx, 0], self.feature_2D[pIdx, 1]), 5, (0, 0, 255), -1)
                            cv2.putText(frame1, str(pIdx), (self.feature_2D[pIdx, 0], self.feature_2D[pIdx, 1]), cv2.FONT_HERSHEY_SIMPLEX, 3.8, (0, 255, 0), 3)

                        self.feature_3D = pixel_to_point(self.color_intrinsics, self.feature_2D, self.depth_image)

                        # Step 7: 将每个 (x, y) 坐标对转换为整数 #
                        cv2.putText(frame1, "id: " + str(markerIds[i]), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Step 8: 计算ArUCo中心 (x, y) 坐标, and 绘制ArUco的中心坐标 #
                        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                        cv2.circle(frame1, (cX, cY), 5, (0, 0, 255), -1)
                        
                    
                    dataTrans = self.feature_2D.reshape(1, -1).squeeze(axis=0)
                    self.pub1.publish(dataTrans)    
                    
                    dataTrans = xyzrgb2pointcloud2(self.feature_3D, self.feature_2D, self.color_image, True, [217, 83, 25], 'world')
                    self.pub2.publish(dataTrans)

                else:
                    print("Marker Not Found!")

                cv2.imshow('frame1', frame1)
                if cv2.waitKey(1) * 0xff == ord('q'):
                    cv2.destroyAllWindows()
                    break
            except BaseException:
                pass


if __name__ == '__main__':
    rospy.init_node('imgpro_calibration', anonymous=True)
    ip = ImageProcessing()
    ip.run()

