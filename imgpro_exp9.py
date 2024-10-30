#! /usr/bin/env python3
import sys
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
sys.path.append('/home/tomqi/Documents/exps_ws/src/exp_m6/script/manipulation (new)')
import cv2
import numpy as np
from General_method2 import *
from ConvertPointcloud import *
from plugins.msg import msg_int, msg_float
from copy import deepcopy
from imgpro_roslaunch import *
from cameraClass import LowPassFilter
from algorithms_pro import *


def find_bag_rim(origin, color, fixedNum=30, base=[247, 59], cropsize=None, clustering=True):
    closing = find_area(origin, threshold=color, hsv_flag=True)[1]

    if cropsize is not None:
        closing = deepcopy(closing[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

    try:
        contours, hierarchy, = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]

        newbinary = np.zeros_like(closing)

        top_k_idx = np.array(contoursArea).argsort()[::-1][0:10]

        # points = np.empty((1, 2), dtype=np.float)
        # for i in range(np.size(top_k_idx)):
        #     points = np.vstack((points, np.squeeze(contours[top_k_idx[i]], axis=1)))
        # points = np.delete(points, 0, axis=0)

        for i in range(np.size(top_k_idx)):
            idx = top_k_idx[i]
            contour = np.squeeze(contours[idx], axis=1)
            newbinary = cv2.fillPoly(newbinary, [contour], (255))

        # kernel = np.ones((3, 3), np.uint8)
        # opening = cv2.morphologyEx(newbinary, cv2.MORPH_OPEN, kernel)

        # kernel = np.ones((3, 3), np.uint8)
        # newbinary = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow('newbinary', newbinary)

        # if cropsize is not None:
        #     contour = contour + np.array([cropsize[0, 1], cropsize[0, 0]])

        # points = contour

        [x, y] = np.where(newbinary == 255)
        points = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))).astype(np.int)
        if cropsize is not None:
            points = points + np.array([cropsize[0, 1], cropsize[0, 0]])

        if clustering:
            points = ClusteringExtractionPoints('FPS', points, fixedNum=fixedNum).astype(np.int)

        points[:, [0, -1]] = points[:, [-1, 0]]
    except:
        pass

    return points


def find_bag_rect(img, color, fixedNum=30, base=[247, 59], cropsize=None):
    closing = find_area(img, threshold=color, hsv_flag=True)[1]

    if cropsize is not None:
        closing = deepcopy(closing[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(closing, kernel, iterations=2)
    # cv2.imshow('erode', erode)

    try:
        contours, hierarchy, = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]
        contour = contours[int(np.argmax(np.array(contoursArea)))]

        contour = np.squeeze(contour, axis=1)

        newbinary = cv2.fillPoly(np.zeros_like(closing, dtype=np.uint8), [contour], (255))
        kernel = np.ones((5, 5), np.uint8)
        erode = cv2.erode(newbinary, kernel, iterations=3)
        # cv2.imshow('erode', erode)

        thinned = cv2.ximgproc.thinning(erode, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        # cv2.imshow('thinned', thinned)

        points = np.hstack((np.where(thinned == 255)[1].reshape((-1, 1)), np.where(thinned == 255)[0].reshape((-1, 1))))
        points = ClusteringExtractionPoints('KMS', points, fixedNum=5).astype(np.int)
        points = points[np.argsort(points[:, 0]), :]

        # cv2.imshow('newbinary', newbinary)

        if cropsize is not None:
            contour = contour + np.array([cropsize[0, 0], cropsize[0, 1]])
            points = points + np.array([cropsize[0, 0], cropsize[0, 1]])

        contour = ClusteringExtractionPoints('FPS', contour, fixedNum=fixedNum).astype(np.int)
        # points[:, 0:-1] = points[:, -1:0]

    except:
        pass

    return np.array(contour, dtype=np.int), points


class ImageProcessing(RealSenseRosSet):
    def __init__(self):
        super(ImageProcessing, self).__init__()
        self.publish_rate = 30.0
        self.rate = rospy.Rate(self.publish_rate)

        self.pub1 = rospy.Publisher("/rim_2D", msg_int, queue_size=10)
        self.pub2 = rospy.Publisher("/rim_3D", PointCloud2, queue_size=10)

        # self.pub3 = rospy.Publisher("/contour_2D", msg_int, queue_size=10)
        # self.pub4 = rospy.Publisher("/contour_3D", PointCloud2, queue_size=10)

        # self.pub5 = rospy.Publisher("/center_2D", msg_int, queue_size=10)
        # self.pub6 = rospy.Publisher("/center_3D", PointCloud2, queue_size=10)

        # self.pub7 = rospy.Publisher("/marker_2D", msg_int, queue_size=10)
        # self.pub8 = rospy.Publisher("/marker_3D", PointCloud2, queue_size=10)

        self.fixedNum = 20

        self.Ts = np.loadtxt('/home/tomqi/Documents/exps_ws/src/plugins/script/CameraCalibration/tf_base_to_camera2.txt')
        # self.Ts = np.eye(4, dtype=np.float)

        # self.cropsize = np.array([[322, 121], [890, 654]])
        self.cropsize = np.array([[260, 45], [970, 730]])

        self.rim_2D = np.zeros((10, 2), dtype=np.uint8)
        self.rim_3D = np.zeros((10, 3), dtype=np.float)
        self.rim_3D_ = np.zeros((10, 3), dtype=np.float)

        # self.contour_2D = np.zeros((10, 2), dtype=np.uint8)
        # self.contour_3D = np.zeros((10, 3), dtype=np.float)
        # self.contour_3D_ = np.zeros((10, 3), dtype=np.float)

        # self.center_2D = np.zeros((10, 2), dtype=np.uint8)
        # self.center_3D = np.zeros((10, 3), dtype=np.float)
        # self.center_3D_ = np.zeros((10, 3), dtype=np.float)

        # self.marker_2D = np.zeros((10, 2), dtype=np.uint8)
        # self.marker_3D = np.zeros((10, 3), dtype=np.float)
        # self.marker_3D_ = np.zeros((10, 3), dtype=np.float)

        self._rimfilterflag = False

        rospy.sleep(1)

    def run(self):
        k = 0
        while not rospy.is_shutdown():
            frame1 = deepcopy(self.color_image)
            frame2 = deepcopy(self.color_image)

            # ----------------------------------------------- RIM -----------------------------------------------#
            try:
                self.rim_2D = find_bag_rim(frame1, np.array([46, 61, 55]), fixedNum=self.fixedNum, base=[247, 59], cropsize=self.cropsize, clustering=False)

                dataTrans = self.rim_2D.reshape(1, -1).squeeze(axis=0)
                self.pub1.publish(dataTrans)

                for i in range(np.size(self.rim_2D, axis=0)):
                    cv2.circle(frame1, (self.rim_2D[i, 0], self.rim_2D[i, 1]), 3, (0, 200, 10), -1)

                self.rim_3D = pixel_to_point(self.color_intrinsics, self.rim_2D, self.depth_image, fixed_depth=True)
                self.rim_3D = transform_shape(self.rim_3D, self.Ts)

                # --------- MIDDLE VALUE FILTERING --------- #
                # centroid = np.mean(self.rim_3D, axis=0)
                # idx = []
                # for j in range(np.size(self.rim_3D, axis=0)):
                #     scale = np.abs(self.rim_3D[j, 2] / centroid[2])
                #     if 0.90 <= scale <= 1.50:
                #         idx.append(j)
                # self.rim_3D = self.rim_3D[np.array(idx), :]

                # self.rim_3D = determine_optimal_hyperplane(self.rim_3D)[3]

                # centroid = np.mean(self.rim_3D, axis=0)
                # results = ProjectStableConfig_2D(self.rim_3D[:, 0:2])
                # self.rim_3D = results[1]
                # self.rim_3D = np.hstack((self.rim_3D, np.ones((np.size(self.rim_3D, axis=0), 1)) * centroid[2] * 1.0))

                # if not self._rimfilterflag:
                #     self.rim_3D_ = self.rim_3D
                #     self._rimfilterflag = True
                # else:
                #     self.rim_3D = LowPassFilter(self.rim_3D_.reshape((-1, 1)).squeeze(), self.rim_3D.reshape((-1, 1)).squeeze(), 0.3)
                #     self.rim_3D = np.reshape(self.rim_3D, (-1, 3))
                #     self.rim_3D_ = self.rim_3D

                # --------- PUBLISH --------- #
                dataTrans = xyzrgb2pointcloud2(self.rim_3D, [0, 255, 0], 'camera_link')
                self.pub2.publish(dataTrans)

            except BaseException:
                pass
                # dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)
                # self.pub1.publish(dataTrans)

                # dataTrans = xyzrgb2pointcloud2(np.array([0, 0, 0]), [0, 255, 0], 'camera_link')
                # self.pub2.publish(dataTrans)

            # ----------------------------------------------- CONTOUR -----------------------------------------------#
            # try:
            #     self.contour_2D, self.marker_2D = find_bag_rect(frame1, np.array([18, 100, 142]), fixedNum=self.fixedNum, base=[247, 59], cropsize=self.cropsize)
            #     dataTrans = self.contour_2D.reshape(1, -1).squeeze(axis=0)
            #     self.pub3.publish(dataTrans)
            #     dataTrans = self.marker_2D.reshape(1, -1).squeeze(axis=0)
            #     self.pub7.publish(dataTrans)
            #
            #     self.contour_3D = Pixel_to_Point(self.color_intrinsics, self.contour_2D, self.depth_image)
            #     self.contour_3D = transform_shape(self.contour_3D, self.Ts)
            #     dataTrans = xyzrgb2pointcloud2(self.contour_3D, [0, 255, 0], 'camera_link')
            #     self.pub4.publish(dataTrans)
            #
            #     self.marker_3D = Pixel_to_Point(self.color_intrinsics, self.marker_2D, self.depth_image)
            #     self.marker_3D = transform_shape(self.marker_3D, self.Ts)
            #     dataTrans = xyzrgb2pointcloud2(self.marker_3D, [0, 255, 0], 'camera_link')
            #     self.pub8.publish(dataTrans)
            #
            # except BaseException:
            #     dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)
            #     self.pub3.publish(dataTrans)
            #
            #     dataTrans = xyzrgb2pointcloud2(np.array([0, 0, 0]), [0, 255, 0], 'camera_link')
            #     self.pub4.publish(dataTrans)

            # ----------------------------------------------- CENTER -----------------------------------------------#
            # try:
            #     self.center_2D = np.mean(self.contour_2D, axis=0).astype(np.int)
            #
            #     dataTrans = self.center_2D.reshape(1, -1).squeeze(axis=0)
            #     self.pub5.publish(dataTrans)
            #
            #     self.center_3D = Pixel_to_Point(self.color_intrinsics, self.center_2D, self.depth_image).reshape(1, -1)
            #     self.center_3D = transform_shape(self.center_3D, Ts=self.Ts)
            #
            #     if not self._markerfilterFlag:
            #         self.center_3D_ = self.center_3D
            #         self._markerfilterFlag = True
            #     else:
            #         self.center_3D = LowPassFilter(self.center_3D_, self.center_3D, 0.9)
            #         self.center_3D_ = self.center_3D
            #
            #     dataTrans = xyzrgb2pointcloud2(self.center_3D, [0, 255, 0], 'camera_link')
            #     self.pub6.publish(dataTrans)
            #
            #     # print('3D-1: ', self.center_3D)
            # except BaseException:
            #     dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)
            #     self.pub5.publish(dataTrans)
            #
            #     dataTrans = xyzrgb2pointcloud2(np.array([0, 0, 0]), [0, 255, 0], 'camera_link')
            #     self.pub6.publish(dataTrans)

            # ----------------------------------------------- VISUALIZING -----------------------------------------------#
            try:
                pass

                # for i in range(self.fixedNum):
                #     cv2.circle(frame1, (self.contour_2D[i, 0], self.contour_2D[i, 1]), 2, (0, 200, 10), -1)

                # for i in range(np.size(self.marker_2D, axis=0)):
                #     cv2.circle(frame1, (self.marker_2D[i, 0], self.marker_2D[i, 1]), 2, (255, 0, 0), -1)
                #     cv2.putText(frame1, str(i), (self.marker_2D[i, 0], self.marker_2D[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                # cv2.circle(frame1, (self.center_2D[0], self.center_2D[1]), 5, (0, 200, 10), -1)

            except BaseException:
                pass

            frame1 = frame1[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.imshow('frame1', frame1)

            # cv2.rectangle(frame2, self.cropsize[0, :], self.cropsize[1, :], (0, 255, 0), 2)
            # cv2.imshow('frame2', frame2)

            if cv2.waitKey(1) * 0xff == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    rospy.init_node('imgpro_fabric', anonymous=True)
    ip = ImageProcessing()
    ip.run()
