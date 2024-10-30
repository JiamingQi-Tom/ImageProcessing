#! /usr/bin/env python3
import sys
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script/image_processing')
import rospy
import message_filters
import json
import pyrealsense2 as rs
import cv2
from General_method1 import *
from General_method2 import *
from ConvertPointcloud import *
from plugins.msg import msg_int, msg_float
from time import time
from cameraClass import *
# from imgpro_exp10 import Obtain3DShape


class ImageProcessing(RealSenseD455Set):
    def __init__(self):
        super(ImageProcessing, self).__init__(displaysize='big')

        self.pub1 = rospy.Publisher("/real_marker_2D", msg_int, queue_size=2)
        self.pub2 = rospy.Publisher("/real_marker_3D", PointCloud2, queue_size=2)

        self.pub3 = rospy.Publisher("/shape_2D", msg_int, queue_size=2)
        self.pub4 = rospy.Publisher("/shape_3D", PointCloud2, queue_size=2)

        self.fixedNum = 64

        self.cropsize = np.array([[296, 60],
                                  [927, 670]])

        self.shapefilter_flag = False

        self.camera_calibration_flag = False
        if self.camera_calibration_flag:
            self.Ts = np.loadtxt(cpath + '/tf_base_to_camera.txt')
        else:
            self.Ts = np.eye(4)

    def run(self):
        while not rospy.is_shutdown():
            color_image, depth_image, aligned_color_frame, aligned_depth_frame = self.get_aligned_images()
            frame1 = deepcopy(color_image)
            frame2 = deepcopy(color_image)

            # ---- Marker Detection ---- #
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
            sel = 1
            try:
                if sel == 1:
                    shape_2D = find_blue_centerline(frame1, color=[97, 110, 64], fixedNum=self.fixedNum, base=[313, 589] + self.cropsize[0, :], cropsize=self.cropsize)[0]
                    # shape_2D = find_blue_centerline(frame1, color=[81, 154, 48], fixedNum=self.fixedNum, base=[313, 589] + self.cropsize[0, :], cropsize=self.cropsize)[0]

                dataTrans = shape_2D.reshape(1, -1).squeeze(axis=0)
                self.pub3.publish(dataTrans)

                shape_3D = Obtain3DShape(shape_2D, aligned_depth_frame, self.depth_intrin)[0]

                shape_3D = transform_shape(shape_3D, self.Ts)

                pointcloud_color = np.zeros((self.fixedNum, 3), dtype=np.uint8)
                for i in range(self.fixedNum):
                    pointcloud_color[i, 0] = np.array([frame1[shape_2D[i, 1], shape_2D[i, 1]]])[0][2]
                    pointcloud_color[i, 1] = np.array([frame1[shape_2D[i, 1], shape_2D[i, 0]]])[0][1]
                    pointcloud_color[i, 2] = np.array([frame1[shape_2D[i, 1], shape_2D[i, 0]]])[0][0]

                dataTrans = xyzrgb2pointcloud2(shape_3D, pointcloud_color, frame_id='camera_link')
                self.pub4.publish(dataTrans)

                for i in range(self.fixedNum):
                    cv2.circle(frame1, (shape_2D[i, 0], shape_2D[i, 1]), 3, (0, 0, 255), -1)
                    # cv2.putText(frame1, str(i), (shape_2D[i, 0], shape_2D[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)

            except BaseException:
                dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)
                self.pub3.publish(dataTrans)

                dataTrans = xyzrgb2pointcloud2(np.array([0, 0, 0]), [255, 0, 0], frame_id='camera_link')
                self.pub4.publish(dataTrans)

            cv2.rectangle(color_image, self.cropsize[0, :], self.cropsize[1, :], (0, 255, 0), 2)
            cv2.imshow('color_image', color_image)

            frame1 = frame1[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.imshow('frame1', frame1)

            frame2 = frame2[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.imshow('frame2', frame2)

            cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('imgpro_exp5', anonymous=True)
    ip = ImageProcessing()
    ip.run()

# self.pub5 = rospy.Publisher("/shapefeature", msg_float, queue_size=1)
# self.pub6 = rospy.Publisher("/fitting_3D", PointCloud2, queue_size=2)

# self.reconstruction_2D = zeros((64, 2), dtype=int)
# self.sub1 = rospy.Subscriber("/desired_2D", msg_int, self.callback1)


# try:
#     left_purple, right_purple, purple_area = find_double_purple(frame1, [117, 56, 68])
#
#     left_yellow, right_yellow, yellow_area = find_double_yellow(frame1, left_purple, right_purple, [8, 91, 121])
#     real_marker_2D = vstack((left_yellow, left_purple, right_purple, right_yellow))
#     real_marker_3D = zeros((size(real_marker_2D, 0), 3), dtype=float)
#     for i in range(size(real_marker_2D, 0)):
#         depth = aligned_depth_frame.get_distance(real_marker_2D[i, 0], real_marker_2D[i, 1])
#         real_marker_3D[i, :] = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [real_marker_2D[i, 0], real_marker_2D[i, 1]], depth)
#
#     if init_flag1 == False:
#         real_marker_3D_previous = real_marker_3D
#         init_flag1 = True
#     else:
#         real_marker_3D = LowPassFIlter(real_marker_3D_previous, real_marker_3D, 0.4)
#         real_marker_3D_previous = real_marker_3D
#
#     dataTrans = asarray(real_marker_2D).reshape(1, -1).squeeze(axis=0)
#     self.pub1.publish(dataTrans)
#
#     dataTrans = xyzrgb2pointcloud2(real_marker_3D, [255, 0, 0], 'world')
#     self.pub2.publish(dataTrans)
#
# except BaseException:
#     pass


# elif sel == 2:
#     # shape_2D = find_pillow(frame1, color=[76, 16, 19], fixedNum=50, start=[left_purple[1], left_purple[0]])
#     # shape_2D = find_pillow(frame1, color=[69, 139, 38], fixedNum=64, start=[left_purple[1], left_purple[0]])
#     # shape_2D = find_pillow(frame1, color=[72, 48, 14], fixedNum=50, start=[left_purple[1], left_purple[0]])
#     # shape_2D = find_pillow(frame1, color=[80, 122, 65], fixedNum=50, start=[left_purple[1], left_purple[0]])
#     # shape_2D = find_pillow(frame1, color=[10, 101, 135], fixedNum=64, start=[left_purple[1], left_purple[0]])
#     # shape_2D = find_contour_middle(frame1, color=[4, 65, 39], fixedNum=64, start=[left_purple[1], left_purple[0]])
#     # shape_2D = find_contour_middle(frame1, color=[80, 122, 65], fixedNum=64, start=[left_purple[1], left_purple[0]])
#     pass
#
# elif sel == 3:
#     # shape_2D = find_yellow_surface(frame1, color=[17, 121, 83], fixedNum=27)
#     # shape_2D = find_yellow_surface(frame1, color=[13, 88, 133], fixedNum=32)
#     pass


# try:
#     for i in range(size(real_marker_2D, 0)):
#         cv2.circle(frame1, (real_marker_2D[i, 0], real_marker_2D[i, 1]), 5, (0, 0, 255), -1)
#         cv2.putText(frame1, '%d' % i, (real_marker_2D[i, 0] - 8, real_marker_2D[i, 1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
# except BaseException:
#     pass

# try:
#     for i in range(size(shape_2D, 0)):
#         cv2.circle(frame2, (shape_2D[i, 1], shape_2D[i, 0]), 3, (0, 0, 255), -1)
#
#     if sel == 1:
#         cv2.putText(frame2, '0', (shape_2D[0, 1] - 2, shape_2D[0, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         # cv2.putText(frame2, '15', (shape_2D[15, 1] - 2, shape_2D[15, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '31', (shape_2D[31, 1] - 2, shape_2D[31, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         # cv2.putText(frame2, '63', (shape_2D[63, 1] - 10, shape_2D[63, 0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#     elif sel == 2:
#         cv2.putText(frame2, '0', (shape_2D[0, 1] - 5, shape_2D[0, 0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '17', (shape_2D[17, 1] - 0, shape_2D[17, 0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '30', (shape_2D[30, 1] + 5, shape_2D[30, 0] - 0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '45', (shape_2D[47, 1] - 0, shape_2D[47, 0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '63', (shape_2D[63, 1] - 30, shape_2D[63, 0] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#     elif sel == 3:
#         cv2.putText(frame2, '0', (shape_2D[0, 1] - 5, shape_2D[0, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '8', (shape_2D[8, 1] - 2, shape_2D[8, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '17', (shape_2D[17, 1] - 0, shape_2D[17, 0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '26', (shape_2D[26, 1] - 13, shape_2D[26, 0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#         cv2.putText(frame2, '31', (shape_2D[31, 1] - 13, shape_2D[31, 0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
#
# except BaseException:
#     pass

# try:
#     if norm(self.reconstruction_2D, 2) > 3:
#         for i in range(size(self.reconstruction_2D, 0)):
#             cv2.circle(frame2, (self.reconstruction_2D[i, 1], self.reconstruction_2D[i, 0]), 5, (10, 255, 0), -1)
#     else:
#         cv2.circle(frame2, (0, 0), 1, (0, 0, 0), -1)
# except BaseException:
#     pass

# try:
#     cv2.putText(frame1, '%.5f' % error, (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_4)
#     cv2.putText(frame1, '%d Hz' % (1.0 / (time() - tic)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_4)
# except BaseException:
#     pass


    # def callback1(self, msg):
    #     temp = asarray(msg.msg_int).astype(int).reshape(-1, 2)
    #     self.reconstruction_2D = temp