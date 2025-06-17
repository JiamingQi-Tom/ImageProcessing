#! /usr/bin/env python3
import sys
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
# import pyrealsense2 as rs
import cv2
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from plugins.msg import msg_float


# def pixel_to_point_fixed_depth(intrinsics, pixel, depth_image, fixed_depth=False):
#     fx = intrinsics[0, 0]
#     fy = intrinsics[1, 1]
#     ppx = intrinsics[0, 2]
#     ppy = intrinsics[1, 2]
#     scale = intrinsics[0, 1]
#
#     if len(pixel) == 2:
#         z = depth_image[pixel[1], pixel[0]] * 0.001
#         x = (pixel[0] - ppx) / fx * z
#         y = (pixel[1] - ppy) / fy * z
#
#         points = np.array([x, y, z], dtype=np.float64)
#
#     else:
#         num = np.size(pixel, axis=0)
#
#         x = np.zeros(num, dtype=np.float64)
#         y = np.zeros(num, dtype=np.float64)
#         z = np.zeros(num, dtype=np.float64)
#
#         if not fixed_depth:
#             z = depth_image[pixel[:, 1], pixel[:, 0]] * 0.001
#             x = (pixel[:, 0] - ppx) / fx * z
#             y = (pixel[:, 1] - ppy) / fy * z
#
#         else:
#             for i in range(num):
#                 z[i] = depth_image[pixel[i, 1], pixel[i, 0]] * 0.001
#
#             idx = np.argmin(np.abs(z - np.mean(z)))
#             depth_fixed = z[idx]
#
#             for i in range(num):
#                 z[i] = depth_fixed
#                 x[i] = (pixel[i, 0] - ppx) / fx * depth_fixed
#                 y[i] = (pixel[i, 1] - ppy) / fy * depth_fixed
#
#         points = np.transpose(np.hstack((x, y, z)).reshape(3, -1))
#
#         zero_idx = np.where(points[:, 2] == 0)[0]
#         nonzero_idx = np.where(points[:, 2] != 0)[0]
#         points[zero_idx, :] = points[nonzero_idx[0], :]
#
#     return points
#

def pixel_to_point(intrinsics, pixel, depth_image):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    ppx = intrinsics[0, 2]
    ppy = intrinsics[1, 2]
    scale = intrinsics[0, 1]

    if len(pixel) == 2:
        z = depth_image[pixel[1], pixel[0]] * 0.001
        x = (pixel[0] - ppx) / fx * z
        y = (pixel[1] - ppy) / fy * z

        points = np.array([x, y, z], dtype=np.float64)

    else:
        z = depth_image[pixel[:, 1], pixel[:, 0]] * 0.001
        x = (pixel[:, 0] - ppx) / fx * z
        y = (pixel[:, 1] - ppy) / fy * z

        points = np.transpose(np.hstack((x, y, z)).reshape(3, -1))

        zero_idx = np.where(points[:, 2] == 0)[0]
        nonzero_idx = np.where(points[:, 2] != 0)[0]
        points[zero_idx, :] = points[nonzero_idx[0], :]

    return points


# def Obtain3DShape(shape_2D, aligned_depth_frame, depth_intrin):
#     depth = np.zeros(np.size(shape_2D, axis=0), dtype=np.float64)
#     shape_3D = np.zeros((np.size(shape_2D, axis=0), 3), dtype=np.float64)

#     for i in range(np.size(shape_2D, axis=0)):
#         depth[i] = aligned_depth_frame.get_distance(shape_2D[i, 0], shape_2D[i, 1])
#         shape_3D[i, :] = rs.rs2_deproject_pixel_to_point(depth_intrin, [shape_2D[i, 0], shape_2D[i, 1]], depth[i])

#     zero_idx = np.where(shape_3D[:, 2] == 0)[0]
#     nonzero_idx = np.where(shape_3D[:, 2] != 0)[0]
#     shape_3D[zero_idx, :] = shape_3D[nonzero_idx[0], :]

#     return shape_3D, depth


class RealSenseRosSet:
    def __init__(self):
        self.color_image = None
        self.depth_image = None
        self.color_image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_image_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.sync1 = message_filters.ApproximateTimeSynchronizer([self.color_image_sub, self.depth_image_sub], 10, 1, allow_headerless=True)
        self.sync1.registerCallback(self.callback1)

        self.color_camera_info = None
        self.depth_camera_info = None
        self.color_camera_info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        self.depth_camera_info_sub = message_filters.Subscriber('/camera/depth/camera_info', CameraInfo)
        self.sync2 = message_filters.ApproximateTimeSynchronizer([self.color_camera_info_sub, self.depth_camera_info_sub], 10, 1, allow_headerless=True)
        self.sync2.registerCallback(self.callback2)

        self.color_intrinsics = np.eye(3, dtype=np.float64)
        self.depth_intrinsics = np.eye(3, dtype=np.float64)

        self.bridge = CvBridge()

        self.mouse_click_position = np.array([0, 0, 0], dtype=np.float64)
        self.pub_mouse_click_position = rospy.Publisher("/mouse_click_position", msg_float, queue_size=2)

        cv2.namedWindow("frame1")
        cv2.setMouseCallback("frame1", self.capture_event)

        rospy.sleep(1)

    def run(self):
        while not rospy.is_shutdown():
            cv2.imshow('frame1', self.color_image)

            # depth_normalized = np.floor(((self.depth_image / np.max(self.depth_image)) * 255)).astype(np.uint8)
            # depth_normalized = cv2.convertScaleAbs(self.depth_image, alpha=255.0 / self.depth_image.max())
            # cv2.imshow('frame1', depth_normalized)
            # print(np.shape(self.color_image))
            # print(np.shape(self.depth_image))

            if cv2.waitKey(1) * 0xff == ord('q'):
                cv2.destroyAllWindows()
                break

    def callback1(self, sub1_message, sub2_message):
        assert isinstance(sub1_message, Image)
        self.color_image = self.bridge.imgmsg_to_cv2(sub1_message, "bgr8")

        assert isinstance(sub2_message, Image)
        self.depth_image = self.bridge.imgmsg_to_cv2(sub2_message, "16UC1")

    def callback2(self, sub1_message, sub2_message):
        assert isinstance(sub1_message, CameraInfo)
        self.color_camera_info = sub1_message
        self.color_intrinsics = np.asarray(self.color_camera_info.K).reshape(3, 3)

        assert isinstance(sub2_message, CameraInfo)
        self.depth_camera_info = sub2_message
        self.depth_intrinsics = np.asarray(self.depth_camera_info.K).reshape(3, 3)

        # self.intrinsics = rs2.intrinsics()
        # self.intrinsics.width = sub1_message.width
        # self.intrinsics.height = sub1_message.height
        # self.intrinsics.ppx = sub1_message.K[2]
        # self.intrinsics.ppy = sub1_message.K[5]
        # self.intrinsics.fx = sub1_message.K[0]
        # self.intrinsics.fy = sub1_message.K[4]
        #
        # if sub1_message.distortion_model == 'plumb_bob':
        #     self.intrinsics.model = rs2.distortion.brown_conrady
        # elif sub1_message.distortion_model == 'equidistant':
        #     self.intrinsics.model = rs2.distortion.kannala_brandt4
        #
        # self.intrinsics.coeffs = [i for i in sub1_message.D]

    def capture_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click_position = pixel_to_point(self.color_intrinsics, [int(x), int(y)], self.depth_image)
            print('2D: (%d, %d)' % (x, y), '  ', end='')
            print('3D: (%2.5f, %2.5f, %2.5f)' % (self.mouse_click_position[0], self.mouse_click_position[1], self.mouse_click_position[2]))

            dataTrans = self.mouse_click_position.reshape(1, -1).squeeze(axis=0)
            self.pub_mouse_click_position.publish(dataTrans)


if __name__ == '__main__':
    rospy.init_node('RealSenseRosSet', anonymous=True)
    ip = RealSenseRosSet()
    ip.run()
