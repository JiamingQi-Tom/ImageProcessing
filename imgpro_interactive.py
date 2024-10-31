#! /usr/bin/env python
import rospy
import message_filters
import json
import pyrealsense2 as rs
from method import *
from General_Image import *
from ConvertPointcloud import *
from experiment1.msg import msg_int, msg_float
from time import time


class ImageProcessing:
    def __init__(self):
        self._to = time()
        rate = rospy.Rate(100)

        self.pub1 = rospy.Publisher("/real_marker_2D", msg_int, queue_size=2)
        self.pub2 = rospy.Publisher("/real_marker_3D", PointCloud2, queue_size=2)

        self.pub3 = rospy.Publisher("/shape_2D", msg_int, queue_size=2)
        self.pub4 = rospy.Publisher("/shape_3D", PointCloud2, queue_size=2)

        self.before_desired_2D = zeros((32, 2), dtype=int)
        self.after_desired_2D = zeros((32, 2), dtype=int)
        self.sub1 = rospy.Subscriber("/before_desired_2D", msg_int, self.callback1)
        self.sub2 = rospy.Subscriber("/after_desired_2D", msg_int, self.callback2)

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
        Ts = loadtxt('/home/qjm/Documents/ur5_ws/src/ur5_planning/src/utlis/src/calibration/Anti-manipulation/3D/transformation.txt')

        while not rospy.is_shutdown():
            tic = time()
            color_image, depth_image, aligned_depth_frame = self.get_aligned_images()
            frame1 = deepcopy(color_image)
            frame2 = deepcopy(color_image)

            # Find real_marker
            try:
                left_purple, right_purple, purple_area = find_double_purple(frame1, [113, 56, 68])
                left_yellow, right_yellow, yellow_area = find_double_yellow(frame1, left_purple, right_purple, [0, 147, 74])
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

                dataTrans = xyzrgb2pointcloud2(real_marker_3D, [255, 30, 30], frame_id='world')
                self.pub2.publish(dataTrans)

            except BaseException:
                pass

            # sel = 1 centerline | sel = 2 contour | sel =3 surface
            try:
                shape_2D = find_blue_centerline(frame1, color=[80, 122, 65], fixedNum=32, start=[left_purple[1], left_purple[0]])[0]
                shape_3D = zeros((size(shape_2D, 0), 3), dtype=float)
                depth = zeros(size(shape_2D, 0), dtype=float)
                for i in range(size(shape_2D, 0)):
                    depth[i] = aligned_depth_frame.get_distance(shape_2D[i, 1], shape_2D[i, 0])
                    shape_3D[i, :] = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [shape_2D[i, 1], shape_2D[i, 0]], depth[i])

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

                # shape_3D = transform_shape(shape_3D, Ts)
                dataTrans = xyzrgb2pointcloud2(shape_3D, [0, 255, 0], 'world')
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

                cv2.putText(frame2, '0', (shape_2D[0, 1] - 2, shape_2D[0, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                cv2.putText(frame2, '15', (shape_2D[15, 1] - 2, shape_2D[15, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                cv2.putText(frame2, '31', (shape_2D[31, 1] - 2, shape_2D[31, 0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)

            except BaseException:
                pass

            try:
                if norm(self.before_desired_2D, 2) > 3:
                    for i in range(size(self.before_desired_2D, 0)):
                        cv2.circle(frame2, (self.before_desired_2D[i, 1], self.before_desired_2D[i, 0]), 3, (10, 255, 0), -1)
                else:
                    cv2.circle(frame2, (0, 0), 1, (0, 0, 0), -1)

                if norm(self.after_desired_2D, 2) > 3:
                    for i in range(size(self.after_desired_2D, 0)):
                        cv2.circle(frame2, (self.after_desired_2D[i, 1], self.after_desired_2D[i, 0]), 3, (222, 10, 10), -1)
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
        self.before_desired_2D = temp

    def callback2(self, msg):
        temp = asarray(msg.msg_int).astype(int).reshape(-1, 2)
        self.after_desired_2D = temp

if __name__ == '__main__':
    rospy.init_node('image_processing', anonymous=True)
    ip = ImageProcessing()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
