#! /usr/bin/env python3
import sys
import cv2
sys.path.append('/home/qjm/Documents/experiments_ws/src/pluginmodules/src')
sys.path.append('/home/qjm/Documents/experiments_ws/src/pluginmodules/src/feature_extraction')
sys.path.append('/home/qjm/Documents/experiments_ws/src/pluginmodules/src/feature_extraction/paper4')
import rospy
import message_filters
import json
import pyrealsense2 as rs
import numpy as np
from experiment4.msg import msg_int, msg_float
from time import time
from General_method1 import *
from General_method2 import *
from ConvertPointcloud import *


class ImageProcessing:
    def __init__(self):
        self._to = time()
        rate = rospy.Rate(100)

        self.pub1 = rospy.Publisher("/real_marker_2D", msg_int, queue_size=2)
        self.pub2 = rospy.Publisher("/real_marker_3D", PointCloud2, queue_size=2)

        self.pub3 = rospy.Publisher("/shape_2D", msg_int, queue_size=2)
        self.pub4 = rospy.Publisher("/shape_3D", PointCloud2, queue_size=2)

        self.reconstruction_2D = zeros((64, 2), dtype=int)
        self.sub1 = rospy.Subscriber("/desired_2D", msg_int, self.callback1)

        # Depth 640x360 1280x720
        # Color 640x480 1280x800
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.intrinsic_get = False

        init_flag1 = False
        init_flag2 = False
        # Ts = np.loadtxt('/home/qjm/Documents/ur5_ws/src/ur5_planning/src/utlis/src/calibration/Anti-manipulation/3D/transformation.txt')
        self.Ts = loadtxt('/home/qjm/Documents/experiments_ws/src/pluginmodules/src/camera_calibration/transformation2.txt')
        while not rospy.is_shutdown():
            tic = time()
            color_image, depth_image, aligned_depth_frame = self.get_aligned_images()
            frame1 = deepcopy(color_image)
            frame2 = deepcopy(color_image)
            frame3 = deepcopy(color_image)

            # Find real_marker
            try:
                left_purple, right_purple, purple_area = find_double_purple(frame1, [113, 56, 68])
                left_yellow, right_yellow, yellow_area = find_double_yellow(frame1, left_purple, right_purple, [0, 147, 80])

                real_marker_2D = vstack((left_yellow, left_purple, right_purple, right_yellow))
                real_marker_3D = zeros((size(real_marker_2D, 0), 3), dtype=float)

                for i in range(size(real_marker_2D, 0)):
                    depth = aligned_depth_frame.get_distance(real_marker_2D[i, 0], real_marker_2D[i, 1])
                    real_marker_3D[i, :] = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [real_marker_2D[i, 0], real_marker_2D[i, 1]], depth)

                if init_flag1 == False:
                    real_marker_3D_previous = real_marker_3D
                    init_flag1 = True
                else:
                    real_marker_3D = LowPassFIlter(real_marker_3D_previous, real_marker_3D, 0.4)
                    real_marker_3D_previous = real_marker_3D

                dataTrans = asarray(real_marker_2D).reshape(1, -1).squeeze(axis=0)
                self.pub1.publish(dataTrans)

                real_marker_3D = transform_shape(real_marker_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(real_marker_3D, [255, 30, 30], frame_id='world')
                self.pub2.publish(dataTrans)

            except BaseException:
                pass

            # sel = 1 centerline | sel = 2 contour | sel = 3 surface
            sel = 1
            try:
                if sel == 1:
                    shape_2D = find_blue_centerline(frame1, color=[80, 122, 65], fixedNum=64, start=[left_purple[1], left_purple[0]])[0]

                elif sel == 2:
                    # shape_2D = find_pillow(frame1, color=[76, 16, 19], fixedNum=50, start=[left_purple[1], left_purple[0]])
                    # shape_2D = find_pillow(frame1, color=[69, 139, 38], fixedNum=64, start=[left_purple[1], left_purple[0]])
                    # shape_2D = find_pillow(frame1, color=[72, 48, 14], fixedNum=50, start=[left_purple[1], left_purple[0]])
                    # shape_2D = find_pillow(frame1, color=[80, 122, 65], fixedNum=50, start=[left_purple[1], left_purple[0]])
                    # shape_2D = find_pillow(frame1, color=[10, 101, 135], fixedNum=64, start=[left_purple[1], left_purple[0]])
                    # shape_2D = find_contour_middle(frame1, color=[4, 65, 39], fixedNum=64, start=[left_purple[1], left_purple[0]])
                    # shape_2D = find_contour_middle(frame1, color=[80, 122, 65], fixedNum=64, start=[left_purple[1], left_purple[0]])
                    # shape_2D = find_contour_black(frame1, color=[80, 122, 65], fixedNum=32, start=[left_purple[1], left_purple[0]])
                    shape_2D = find_contour_middle1(frame1, color=[49, 97, 88], fixedNum=64, start=[left_purple[1], left_purple[0]])

                elif sel == 3:
                    # shape_2D = find_yellow_surface(frame1, color=[8, 82, 132], fixedNum=27, kernel_size=3)
                    # shape_2D = find_yellow_surface(frame1, color=[12, 60, 79], fixedNum=27, kernel_size=3)
                    # shape_2D = find_yellow_surface(frame1, color=[13, 88, 133], fixedNum=32)
                    shape_2D = find_yellow_surface(frame1, color=[14, 118, 129], fixedNum=32)

                shape_3D = zeros((size(shape_2D, 0), 3), dtype=float)
                depth = zeros(size(shape_2D, 0), dtype=float)
                for i in range(size(shape_2D, 0)):
                    depth[i] = aligned_depth_frame.get_distance(shape_2D[i, 1], shape_2D[i, 0])
                    shape_3D[i, :] = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [shape_2D[i, 1], shape_2D[i, 0]], depth[i])

                # print shape_3D
                if init_flag2 == False:
                    shape_3D_previous = shape_3D
                    init_flag2 = True
                else:
                    shape_3D = LowPassFIlter(shape_3D_previous, shape_3D, 1)
                    shape_3D_previous = shape_3D

                zero_idx = where(shape_3D_previous[:, 2] == 0)[0]
                nonzero_idx = where(shape_3D_previous[:, 2] != 0)[0]
                shape_3D_previous[zero_idx, :] = shape_3D_previous[nonzero_idx[0], :]

                dataTrans = shape_2D.reshape(1, -1).squeeze(axis=0)
                self.pub3.publish(dataTrans)

                shape_3D = transform_shape(shape_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(shape_3D, [0, 255, 0], frame_id='world')
                self.pub4.publish(dataTrans)
            except BaseException:
                pass

            # Plot marker, centerline, contour, and surface
            try:
                for i in range(size(real_marker_2D, 0)):
                    cv2.circle(frame1, (real_marker_2D[i, 0], real_marker_2D[i, 1]), 5, (0, 0, 255), -1)
                    cv2.putText(frame1, '%d' % i, (real_marker_2D[i, 0] - 8, real_marker_2D[i, 1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
            except BaseException:
                pass

            try:
                for i in range(size(shape_2D, 0)):
                    cv2.circle(frame2, (shape_2D[i, 1], shape_2D[i, 0]), 3, (0, 0, 255), -1)

                if sel == 1:
                    # cv2.putText(frame2, '0', (shape_2D[0, 1] - 2, shape_2D[0, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    # cv2.putText(frame2, '15', (shape_2D[15, 1] - 2, shape_2D[15, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    # cv2.putText(frame2, '31', (shape_2D[31, 1] - 2, shape_2D[31, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '63', (shape_2D[63, 1] - 10, shape_2D[63, 0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                elif sel == 2:
                    cv2.putText(frame2, '0', (shape_2D[0, 1] - 25, shape_2D[0, 0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '17', (shape_2D[17, 1] - 0, shape_2D[17, 0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '30', (shape_2D[30, 1] + 5, shape_2D[30, 0] - 0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '45', (shape_2D[47, 1] - 0, shape_2D[47, 0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '63', (shape_2D[63, 1] - 30, shape_2D[63, 0] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                elif sel == 3:
                    cv2.putText(frame2, '0', (shape_2D[0, 1] - 5, shape_2D[0, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '8', (shape_2D[8, 1] - 2, shape_2D[8, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '17', (shape_2D[17, 1] - 0, shape_2D[17, 0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '26', (shape_2D[26, 1] - 13, shape_2D[26, 0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(frame2, '31', (shape_2D[31, 1] - 13, shape_2D[31, 0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)

            except BaseException:
                pass

            try:
                if norm(self.reconstruction_2D, 2) > 3:
                    for i in range(size(self.reconstruction_2D, 0)):
                        cv2.circle(frame2, (self.reconstruction_2D[i, 1], self.reconstruction_2D[i, 0]), 5, (10, 255, 0), -1)
                else:
                    cv2.circle(frame2, (0, 0), 1, (0, 0, 0), -1)
            except BaseException:
                pass

            try:
                cv2.putText(frame1, '%d Hz' % (1.0 / (time() - tic)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_4)
            except BaseException:
                pass

            cv2.imshow('frame1', frame1)
            cv2.imshow('frame2', frame2)
            cv2.imshow('frame3', frame3)
            cv2.waitKey(1)
            rate.sleep()

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        if not self.intrinsic_get:
            self.color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
            self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            camera_parameters = {'fx': self.color_intrin.fx, 'fy': self.color_intrin.fy,
                                 'ppx': self.color_intrin.ppx, 'ppy': self.color_intrin.ppy,
                                 'height': self.color_intrin.height, 'width': self.color_intrin.width,
                                 'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
                                 }
            with open('./intrinsics.json', 'w') as fp:
                json.dump(camera_parameters, fp)

            self.intrinsic_get = True

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        return color_image, depth_image, aligned_depth_frame

    def callback1(self, msg):
        temp = asarray(msg.msg_int).astype(int).reshape(-1, 2)
        self.reconstruction_2D = temp


if __name__ == '__main__':
    rospy.init_node('image_processing', anonymous=True)
    ip = ImageProcessing()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()