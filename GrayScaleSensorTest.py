#! /usr/bin/env python3
import sys
# sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
# sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script/image_processing')
# import rospy
# import message_filters
# import json
import numpy as np
import pyrealsense2 as rs
import cv2
# from General_method1 import *
# from General_method2 import *
# from ConvertPointcloud import *
# from plugins.msg import msg_int, msg_float
from time import time
# from cameraClass import *
# from imgpro_exp10 import Obtain3DShape
from cameraClass_API import *
from imgpro_extraction import *

class ImageProcessing(RealSenseD405Set):
    def __init__(self):
        super(ImageProcessing, self).__init__(displaysize='small')

        # self.pub1 = rospy.Publisher("/real_marker_2D", msg_int, queue_size=2)
        # self.pub2 = rospy.Publisher("/real_marker_3D", PointCloud2, queue_size=2)

        # self.pub3 = rospy.Publisher("/shape_2D", msg_int, queue_size=2)
        # self.pub4 = rospy.Publisher("/shape_3D", PointCloud2, queue_size=2)

        # self.fixedNum = 64

        # self.cropsize = np.array([[296, 60],
        #                           [927, 670]])

        # self.shapefilter_flag = False

        # self.camera_calibration_flag = False
        # if self.camera_calibration_flag:
        #     self.Ts = np.loadtxt(cpath + '/tf_base_to_camera.txt')
        # else:
        #     self.Ts = np.eye(4)

        self.detectionPoints = np.empty((30, 2), dtype=np.int64)
        

        theta = np.linspace(start=0, stop=np.pi * 2, num=30, endpoint=True)
        for i in range(np.size(theta)):
            self.detectionPoints[i, 0] = 150 * np.cos(theta[i]) + 320
            self.detectionPoints[i, 1] = 150 * np.sin(theta[i]) + 240


        # self.detectionPoints = np.array([[320, 240],
        #                                  [320, 280],
        #                                  [320, 200],
        #                                  [360, 240],
        #                                  [280, 240]], dtype=np.int64)

    def run(self):
        while True:
            tic = time()

            left_color_image = self.get_aligned_images(pipeline=self.pipelines["left_D405"])[0]
            right_color_image = self.get_aligned_images(pipeline=self.pipelines["right_D405"])[0]
            frame1 = deepcopy(left_color_image)
            frame2 = deepcopy(right_color_image)
            
            # ---- Marker Detection ---- #

            _, closing1 = extraction_thr(frame1, threshold=80)
            _, closing2 = extraction_thr(frame2, threshold=80)
            
            # thresholded = cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
            # print(thresholded)

                
            detectionFlag1 = closing1[self.detectionPoints[:, 1], self.detectionPoints[:, 0]]
            detectionFlag2 = closing2[self.detectionPoints[:, 1], self.detectionPoints[:, 0]]
                

            for i in range(np.size(detectionFlag1)):
                if detectionFlag1[i] == 255:
                    detectionFlag1[i] = 1
                    cv2.circle(frame1, (int(self.detectionPoints[i, 0]), int(self.detectionPoints[i, 1])), 8, (0, 0, 255), -1)
                else:
                    cv2.circle(frame1, (int(self.detectionPoints[i, 0]), int(self.detectionPoints[i, 1])), 8, (0, 255, 0), -1)
                
                cv2.putText(frame1, str(i), (int(self.detectionPoints[i, 0] + 5), int(self.detectionPoints[i, 1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_4)

            for i in range(np.size(detectionFlag2)):
                if detectionFlag2[i] == 255:
                    detectionFlag2[i] = 1
                    cv2.circle(frame2, (int(self.detectionPoints[i, 0]), int(self.detectionPoints[i, 1])), 8, (0, 0, 255), -1)
                else:
                    cv2.circle(frame2, (int(self.detectionPoints[i, 0]), int(self.detectionPoints[i, 1])), 8, (0, 255, 0), -1)
            
            # ret, thresh = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # threshold_image = cv2.adaptiveThreshold(closing, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            # cv2.imshow('threshold_image', threshold_image)

            # print(detectionFlag)
            cv2.imshow('closing1', closing1)
            cv2.imshow('closing2', closing2)

            
            # try:
            #     # real_marker_2D = find_contour(frame1, color=[23, 60, 111], fixedNum=self.fixedNum, cropsize=self.cropsize)[1]
            #     real_marker_2D = find_contour(frame1, color=[16, 123, 71], fixedNum=self.fixedNum, cropsize=self.cropsize)[1]
            #     real_marker_2D = real_marker_2D.astype(int)
            #
            #     cv2.circle(frame1, (int(real_marker_2D[0]), int(real_marker_2D[1])), 3, (0, 0, 255), -1)
            #
            #     dataTrans = real_marker_2D.reshape(1, -1).squeeze(axis=0)
            #     self.pub1.publish(dataTrans)
            #
            #     real_marker_3D = Obtain3DShape(real_marker_2D.astype(int), aligned_depth_frame, self.depth_intrin)[0]
            #
            #     real_marker_3D = transform_shape(real_marker_3D, self.Ts)
            #     dataTrans = xyzrgb2pointcloud2(real_marker_3D, [255, 0, 0], 'world')
            #     self.pub2.publish(dataTrans)
            #
            # except BaseException:
            #     dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)
            #     self.pub1.publish(dataTrans)
            #
            #     dataTrans = xyzrgb2pointcloud2(np.array([0, 0, 0]), [255, 0, 0], 'world')
            #     self.pub2.publish(dataTrans)

            # ---- Shape Detection ---- #
            # sel = 1
            # try:
            #     if sel == 1:
            #         shape_2D = find_blue_centerline(frame1, color=[97, 110, 64], fixedNum=self.fixedNum, base=[313, 589] + self.cropsize[0, :], cropsize=self.cropsize)[0]
            #         # shape_2D = find_blue_centerline(frame1, color=[81, 154, 48], fixedNum=self.fixedNum, base=[313, 589] + self.cropsize[0, :], cropsize=self.cropsize)[0]

            #     dataTrans = shape_2D.reshape(1, -1).squeeze(axis=0)
            #     self.pub3.publish(dataTrans)

            #     shape_3D = Obtain3DShape(shape_2D, aligned_depth_frame, self.depth_intrin)[0]

            #     shape_3D = transform_shape(shape_3D, self.Ts)

            #     pointcloud_color = np.zeros((self.fixedNum, 3), dtype=np.uint8)
            #     for i in range(self.fixedNum):
            #         pointcloud_color[i, 0] = np.array([frame1[shape_2D[i, 1], shape_2D[i, 1]]])[0][2]
            #         pointcloud_color[i, 1] = np.array([frame1[shape_2D[i, 1], shape_2D[i, 0]]])[0][1]
            #         pointcloud_color[i, 2] = np.array([frame1[shape_2D[i, 1], shape_2D[i, 0]]])[0][0]

            #     dataTrans = xyzrgb2pointcloud2(shape_3D, pointcloud_color, frame_id='camera_link')
            #     self.pub4.publish(dataTrans)

            #     for i in range(self.fixedNum):
            #         cv2.circle(frame1, (shape_2D[i, 0], shape_2D[i, 1]), 3, (0, 0, 255), -1)
            #         # cv2.putText(frame1, str(i), (shape_2D[i, 0], shape_2D[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)

            # except BaseException:
            #     dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)
            #     self.pub3.publish(dataTrans)

            #     dataTrans = xyzrgb2pointcloud2(np.array([0, 0, 0]), [255, 0, 0], frame_id='camera_link')
            #     self.pub4.publish(dataTrans)

            # cv2.rectangle(color_image, self.cropsize[0, :], self.cropsize[1, :], (0, 255, 0), 2)
            # cv2.imshow('color_image', color_image)

            # frame1 = frame1[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.imshow('frame1', frame1)

            # frame2 = frame2[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.imshow('frame2', frame2)

        

            cv2.waitKey(1)

            # print(1.0 / (time() - tic))


if __name__ == '__main__':
    # rospy.init_node('imgpro_exp5', anonymous=True)
    ip = ImageProcessing()
    ip.run()
