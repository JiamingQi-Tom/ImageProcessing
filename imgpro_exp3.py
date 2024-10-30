#! /usr/bin/env python
import rospy
import time
import json
import numpy as np
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from method import *
from time import time
# from experiment.msg import mymsg, msg_float
# from experiment1.msg import msg_int, msg_float
from math import *

# self.image_pub3 = rospy.Publisher("/left_red_center", mymsg, queue_size=1)
# self.reconstruct_left_red_center = np.zeros((2, 2)).astype(int)
# self.image_sub3 = rospy.Subscriber("/reconstruct_left_red_center", mymsg, self.callback3)


class ImageProcessing:
    def __init__(self):
        self._to = time()
        self.rate = rospy.Rate(50)

        self.image_pub1 = rospy.Publisher("/real_marker", msg_int, queue_size=1)
        self.image_pub2 = rospy.Publisher("/contour", msg_int, queue_size=1)
        self.image_pub3 = rospy.Publisher("/contour_moments", msg_float, queue_size=1)

        self.reconstruct_contour = np.zeros((300, 2)).astype(int)
        self.image_sub1 = rospy.Subscriber("/reconstruct_contour", msg_int, self.callback1)

        # Depth 640x360
        # Color 640x480 1280x720
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.intrinsic_get = False

        while not rospy.is_shutdown():
            tic = time()
            color_image, depth_image, aligned_depth_frame = self.get_aligned_images()
            frame1 = deepcopy(color_image)
            frame2 = deepcopy(color_image)

            # Find real_marker
            try:
                left_purple, right_purple, purple_area = find_double_purple(frame1)
                left_yellow, right_yellow, yellow_area = find_double_yellow(frame1, left_purple, right_purple)
            except BaseException:
                pass

            # Find the contours of rod or sponge
            case = 2
            try:
                if case == 1:
                    contour = find_black_rod_contour(frame1, purple_area, yellow_area, left_purple, right_yellow, fixedNum=300)
                elif case == 2:
                    contour, center = find_yellow_cloth(frame1, [6, 60, 119], left_purple, right_yellow, fixedNum=300)
            except BaseException:
                pass

            # Calculate contour moment
            try:
                contM = paper3_shape_feature(contour)
                print '%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (contM[0], contM[1], contM[2], contM[3], contM[4],
                                                                                      contM[5], contM[6], contM[7], contM[8], contM[9])
            except BaseException:
                pass

            # Plot and label points
            try:
                pass
                # cv2.line(frame1, (319, 0), (319, 479), (0, 255, 0), 1)

                # cv2.circle(frame1, (left_purple[0], left_purple[1]), 3, (0, 255, 0), -1)
                # cv2.circle(frame1, (right_purple[0], right_purple[1]), 3, (0, 255, 0), -1)
                # cv2.circle(frame1, (left_yellow[0], left_yellow[1]), 3, (0, 255, 0), -1)
                # cv2.circle(frame1, (right_yellow[0], right_yellow[1]), 3, (0, 255, 0), -1)

                # cv2.putText(frame1, '1', (left_purple[0] - 12, left_purple[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.putText(frame1, '2', (left_yellow[0] - 12, left_yellow[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.putText(frame1, '3', (right_purple[0] - 12, right_purple[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.putText(frame1, '4', (right_yellow[0] - 12, right_yellow[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

                # cv2.circle(frame2, (center[0], center[1]), 5, (0, 255, 0), -1)
                # cv2.circle(frame2, (left_red[0], left_red[1]), 5, (0, 255, 0), -1)
                # cv2.line(frame2, (left_red[0], left_red[1]), (center[0], center[1]), (255, 0, 0), 2)
                # cv2.putText(frame2, 'Centroid', (center[0] - 38, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(frame2, 'marker1', (left_red[0] - 50, left_red[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

                # for i in range(size(purple_area, 0)):
                #     cv2.circle(frame1, (purple_area[i, 1], purple_area[i, 0]), 3, (0, 255, 0), -1)#
                # for i in range(size(yellow_area, 0)):
                #     cv2.circle(frame1, (yellow_area[i, 1], yellow_area[i, 0]), 3, (255, 0, 0), -1)

            except BaseException:
                pass

            try:
                for i in range(size(contour, 0)):
                    cv2.circle(frame2, (contour[i, 0], contour[i, 1]), 1, (0, 255, 0), -1)
            except BaseException:
                pass

            # Plot reconstruction contour
            try:
                for i in range(np.size(self.reconstruct_contour, 0)):
                    cv2.circle(frame2, (self.reconstruct_contour[i, 0], self.reconstruct_contour[i, 1]), 1, (0, 0, 255), -1)
            except BaseException:
                pass

        # # Plot reconstruction contour
        # # try:
        # #     cv2.line(frame2, (self.reconstruct_left_red_center[0][0], self.reconstruct_left_red_center[0][1]),
        # #              (self.reconstruct_left_red_center[1][0], self.reconstruct_left_red_center[1][1]), (147, 20, 255), 2)
        # #     cv2.circle(frame2, (self.reconstruct_left_red_center[1][0], self.reconstruct_left_red_center[1][1]), 5, (0, 0, 255), -1)
        # # except BaseException:
        # #     pass
        #
            # Publish points
            try:
                dataTrans = np.asarray([left_purple, left_yellow, right_purple, right_yellow]).reshape(1, -1).squeeze(axis=0)
                self.image_pub1.publish(dataTrans)
            except BaseException:
                pass

            # Publish contour
            try:
                dataTrans = contour.reshape(1, -1).squeeze(axis=0)
                self.image_pub2.publish(dataTrans)
            except BaseException:
                pass

            # Publish contour moments
            try:
                dataTrans = contM.reshape(1, -1).squeeze(axis=0)
                self.image_pub3.publish(dataTrans)
            except BaseException:
                pass
        #
        # # Publish points
        # try:
        #     dataTrans = np.asarray([left_red, center]).reshape(1, -1).squeeze(axis=0)
        #     self.image_pub3.publish(dataTrans)
        # except BaseException:
        #     pass
        #

            cv2.imshow('frame1', frame1)
            cv2.imshow('frame2', frame2)
            cv2.waitKey(1)

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        if not self.intrinsic_get:
            self.intr = aligned_color_frame.profile.as_video_stream_profile().intrinsics
            self.depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            camera_parameters = {'fx': self.intr.fx, 'fy': self.intr.fy,
                                 'ppx': self.intr.ppx, 'ppy': self.intr.ppy,
                                 'height': self.intr.height, 'width': self.intr.width,
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
        self.reconstruct_contour = temp

    # def callback3(self, msg):
    #     temp = np.asarray(msg.marker_centerline).astype(int).reshape(-1, 2)
    #     self.reconstruct_left_red_center = temp


if __name__ == '__main__':
    rospy.init_node('image_processing', anonymous=True)
    ip = ImageProcessing()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
