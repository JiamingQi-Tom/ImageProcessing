#! /usr/bin/env python3
# import sys
# sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
import os
cpath = os.path.abspath(os.path.dirname(__file__))
ccpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
import pyrealsense2 as rs
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
import cv2
# from ConvertPointcloud import *
from copy import deepcopy
from time import time


class RealSenseD455Set:
    def __init__(self, displaysize='small'):
        self.ctx = rs.context()
        self.devices = self.ctx.query_devices()
        self.displaysize = displaysize
        self.config = rs.config()
        self.pipelines = {}

        self.camera1_id = 239222302149
        self.camera2_id = 242422304724

        for dev in self.devices:
            dev_name = dev.get_info(rs.camera_info.name)
            if "D455" in dev_name:
                self.pipeline = rs.pipeline(self.ctx)
                self.config = rs.config()
                self.config.enable_device(dev.get_info(rs.camera_info.serial_number))
                print('Serial number: ', dev.get_info(rs.camera_info.serial_number))

                if self.displaysize == 'small':
                    self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
                    self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                elif self.displaysize == 'big':
                    self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                    self.config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
                else:
                    print('Wrong camera size, please re-set!!!')

                self.profile = self.pipeline.start(self.config)

                # self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                # self.depth_sensor = self.profile.get_device().first_depth_sensor()
                # self.depth_sensor.set_option(rs.option.visual_preset, 2)
                # self.depth_sensor.set_option(rs.option.brightness, 2)
                # self.depth_sensor.set_option(rs.option.enable_auto_exposure, True)

                # sensors = dev.query_sensors()
                # for sensor in sensors:
                #     if sensor.is_depth_sensor:
                #         sensor.set_option(rs.option.exposure, 140000.00)
                #         sensor.set_option(rs.option.gain, 39.00)
                #         print(sensor.get_option(rs.option.exposure))

                self.align_to = rs.stream.color
                self.align = rs.align(self.align_to)

                if int(dev.get_info(rs.camera_info.serial_number)) == self.camera1_id:
                    self.pipelines["camera1"] = self.pipeline

                elif int(dev.get_info(rs.camera_info.serial_number)) == self.camera2_id:
                    self.pipelines["camera2"] = self.pipeline

                self.intrinsic_get = False
                self.extrinsic_get = False

        # self.sensor = self.profile.get_device().first_depth_sensor()
        # depth_sensor = self.profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # print(self.sensor.get_option(rs.option.exposure))
        # print(self.sensor.get_option_range(rs.option.exposure))

        # self.align_to = rs.stream.color
        # self.align = rs.align(self.align_to)
        # self.intrinsic_get = False

        # cv2.namedWindow("frame1")
        # cv2.setMouseCallback("frame1", self.capture_event)

    def display(self):
        while True:
            camera1 = self.get_aligned_images(pipeline=self.pipelines["camera1"])[0]
            camera2 = self.get_aligned_images(pipeline=self.pipelines["camera2"])[0]

            cv2.imshow('camera1', camera1)
            cv2.imshow('camera2', camera2)
            # combine = np.hstack((camera1, camera2))
            # cv2.imshow('combine', combine)
            cv2.waitKey(1)

            # (h, w) = camera1.shape[:2]
            # center = (w // 2, h // 2)
            # M = cv2.getRotationMatrix2D(center, 180, 1.0)
            # rotated_image1 = cv2.warpAffine(camera1, M, (w, h))
            # rotated_image2 = cv2.warpAffine(camera2, M, (w, h))
            # cv2.imshow('rotated_image1', rotated_image1)
            # cv2.imshow('rotated_image1', rotated_image2)
            # combine = np.hstack((rotated_image1, rotated_image2))
            # cv2.imshow('combine', combine)
            # cv2.waitKey(1)

    def get_aligned_images(self, pipeline):
        frames = pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        if not self.intrinsic_get:
            self.color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
            self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

            self.intrinsic_get = True

        color_image = np.asanyarray(aligned_color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        return color_image, depth_image, aligned_color_frame, aligned_depth_frame


class RealSenseD405Set:
    """
    @
    @
    @
    """

    def __init__(self, displaysize='small'):
        self.ctx = rs.context()
        self.devices = self.ctx.query_devices()
        self.align = rs.align(rs.stream.color)
        self.depth_scale = None
        self.pipelines = {}

        self.left_camera_id = 130322272491
        self.right_camera_id = 128422270849

        for dev in self.devices:
            dev_name = dev.get_info(rs.camera_info.name)
            if "D405" in dev_name:
                self.pipeline = rs.pipeline(self.ctx)
                self.config = rs.config()
                self.config.enable_device(dev.get_info(rs.camera_info.serial_number))
                # print('Serial number: ', dev.get_info(rs.camera_info.serial_number))

                self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

                self.profile = self.pipeline.start(self.config)
                self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

                self.depth_sensor = self.profile.get_device().first_depth_sensor()

                self.depth_sensor.set_option(rs.option.visual_preset, 2)
                self.depth_sensor.set_option(rs.option.brightness, 2)

                self.depth_sensor.set_option(rs.option.enable_auto_exposure, True)
                # sensors = dev.query_sensors()
                # for sensor in sensors:
                #     if sensor.is_depth_sensor:
                #         sensor.set_option(rs.option.exposure, 140000.00)
                #         sensor.set_option(rs.option.gain, 39.00)
                #         print(sensor.get_option(rs.option.exposure))

                self.align_to = rs.stream.color
                self.align = rs.align(self.align_to)

                if int(dev.get_info(rs.camera_info.serial_number)) == self.left_camera_id:
                    self.pipelines["left_D405"] = self.pipeline

                elif int(dev.get_info(rs.camera_info.serial_number)) == self.right_camera_id:
                    self.pipelines["right_D405"] = self.pipeline

                self.intrinsic_get = False
                self.extrinsic_get = False

        self.Ts = np.eye(4)

        # cv2.namedWindow("D405_color_image")
        # cv2.setMouseCallback("D405_color_image", self.capture_event)

    def display(self):
        while True:
            left_color_image = self.get_aligned_images(pipeline=self.pipelines["left_D405"])[0]
            right_color_image = self.get_aligned_images(pipeline=self.pipelines["right_D405"])[0]

            cv2.imshow('left_color_image', left_color_image)
            cv2.imshow('right_color_image', right_color_image)
            cv2.waitKey(1)

    def get_aligned_images(self, pipeline):
        frames = pipeline.wait_for_frames()
        # frames = self.pipelines["D405"].wait_for_frames()
        aligned_frames = self.align.process(frames)
        # aligned_frames = frames
        aligned_color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        if not self.intrinsic_get:
            self.color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
            self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

            self.intrinsic_get = True

        color_image = np.asanyarray(aligned_color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        return color_image, depth_image, aligned_color_frame, aligned_depth_frame

    def capture_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            color_image, depth_image, aligned_color_frame, aligned_depth_frame = self.get_aligned_images()
            depth = aligned_depth_frame.get_distance(int(x), int(y))
            position = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(x), int(y)], depth)
            position = transform_shape(np.asarray(position).reshape(1, -1), self.Ts)
            print('2D: (%d, %d)' % (x, y),
                  '     ',
                  '3D: (%2.5f, %2.5f %2.5f)' % (position[0], position[1], position[2]))


if __name__ == '__main__':
    ip = RealSenseD455Set(displaysize='big')
    ip.display()

    # ip = RealSenseD405Set(displaysize='small')
    # ip.display()



# def LowPassFilter(y, x, a):
#     output = (1 - a) * y + a * x
#     return output
#
#
# def LKFShapeFilter(x, y, P0, var_s, var_r):
#     dimension = np.size(x, 0) * np.size(x, 1)
#     A = np.eye(dimension)
#     C = np.eye(dimension)
#     xe0 = x.reshape(-1, 1)
#     y = y.reshape(-1, 1)
#
#     P = A.dot(P0).dot(np.transpose(A)) + var_s
#     K = P.dot(np.transpose(C)).dot(np.linalg.inv(C.dot(P).dot(np.transpose(C)) + var_r))
#     P = (np.eye(dimension) - K.dot(C)).dot(P)
#     xe = A.dot((xe0)) + K.dot(y - C.dot(A).dot(xe0))
#     filter = xe.reshape(-1, 3)
#     return filter, P


# class RealSenseD455Set:
#     """
#     @
#     @
#     @
#     """
#     def __init__(self, displaysize='small'):
#         self.displaysize = displaysize
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#
#         if self.displaysize == 'small':
#             self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
#             self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         elif self.displaysize == 'big':
#             self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#             self.config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
#         else:
#             print('Wrong camera size, please re-set!!!')
#
#         self.profile = self.pipeline.start(self.config)
#
#         # self.sensor = self.profile.get_device().first_depth_sensor()
#         # depth_sensor = self.profile.get_device().first_depth_sensor()
#         # depth_scale = depth_sensor.get_depth_scale()
#         # print(self.sensor.get_option(rs.option.exposure))
#         # print(self.sensor.get_option_range(rs.option.exposure))
#
#         self.align_to = rs.stream.color
#         self.align = rs.align(self.align_to)
#         self.intrinsic_get = False
#
#         self.camera_calibration_flag = False
#         if self.camera_calibration_flag:
#             self.Ts = np.loadtxt('/home/tomqi/Documents/exps_ws/src/plugins/script/ImageProcessing/tf_base_to_camera.txt')
#         else:
#             self.Ts = np.eye(4)
#
#         cv2.namedWindow("frame1")
#         cv2.setMouseCallback("frame1", self.capture_event)
#
#     def display(self):
#         while True:
#             color_image = self.get_aligned_images()[0]
#
#             cv2.imshow('frame1', color_image)
#             cv2.waitKey(1)
#
#     def get_aligned_images(self):
#         frames = self.pipeline.wait_for_frames()
#         aligned_frames = self.align.process(frames)
#         aligned_color_frame = aligned_frames.get_color_frame()
#         aligned_depth_frame = aligned_frames.get_depth_frame()
#         if not self.intrinsic_get:
#             self.color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
#             self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
#
#             # camera_parameters = {'fx': self.color_intrin.fx, 'fy': self.color_intrin.fy,
#             #                      'ppx': self.color_intrin.ppx, 'ppy': self.color_intrin.ppy,
#             #                      'height': self.color_intrin.height, 'width': self.color_intrin.width,
#             #                      'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
#             #                      }
#             # with open('/home/tomqi/Documents/exps_ws/src/plugin/srcipt/ImageProcessing/intrinsics.json', 'w') as fp:
#             #     json.dump(camera_parameters, fp)
#
#             self.intrinsic_get = True
#
#         color_image = np.asanyarray(aligned_color_frame.get_data())
#         depth_image = np.asanyarray(aligned_depth_frame.get_data())
#         return color_image, depth_image, aligned_color_frame, aligned_depth_frame
#
#     def capture_event(self, event, x, y, flags, params):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             color_image, depth_image, aligned_color_frame, aligned_depth_frame = self.get_aligned_images()
#             depth = aligned_depth_frame.get_distance(int(x), int(y))
#             position = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(x), int(y)], depth)
#             position = transform_shape(np.asarray(position).reshape(1, -1), self.Ts)
#             print('2D: (%d, %d)' % (x, y),
#                   '     ',
#                   '3D: (%2.5f, %2.5f %2.5f)' % (position[0], position[1], position[2]))
