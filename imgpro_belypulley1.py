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


class RealSenseRosSet:
    def __init__(self):
        self.color_image1 = None
        self.depth_image1 = None
        
        self.color_image2 = None
        self.depth_image2 = None


        self.color_image_sub1 = message_filters.Subscriber('/camera1/color/image_raw', Image)
        self.depth_image_sub1 = message_filters.Subscriber('/camera1/aligned_depth_to_color/image_raw', Image)
        
        self.color_image_sub2 = message_filters.Subscriber('/camera2/color/image_raw', Image)
        self.depth_image_sub2 = message_filters.Subscriber('/camera2/aligned_depth_to_color/image_raw', Image)


        self.sync1 = message_filters.ApproximateTimeSynchronizer([self.color_image_sub1, self.depth_image_sub1,
                                                                  self.color_image_sub2, self.depth_image_sub2], 10, 1, allow_headerless=True)
        self.sync1.registerCallback(self.callback1)
        
        # self.sync2 = message_filters.ApproximateTimeSynchronizer([self.color_image_sub2, self.depth_image_sub2], 10, 1, allow_headerless=True)
        # self.sync2.registerCallback(self.callback2)


        # self.color_camera_info = None
        # self.depth_camera_info = None
        # self.color_camera_info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        # self.depth_camera_info_sub = message_filters.Subscriber('/camera/depth/camera_info', CameraInfo)
        # self.sync2 = message_filters.ApproximateTimeSynchronizer([self.color_camera_info_sub, self.depth_camera_info_sub], 10, 1, allow_headerless=True)
        # self.sync2.registerCallback(self.callback2)

        # self.color_intrinsics = np.eye(3, dtype=np.float64)
        # self.depth_intrinsics = np.eye(3, dtype=np.float64)

        self.bridge = CvBridge()

        # self.mouse_click_position = np.array([0, 0, 0], dtype=np.float64)
        # self.pub_mouse_click_position = rospy.Publisher("/mouse_click_position", msg_float, queue_size=2)

        # cv2.namedWindow("frame1")
        # cv2.setMouseCallback("frame1", self.capture_event)

        rospy.sleep(1)

    def run(self):
        while not rospy.is_shutdown():
            try:
                cv2.imshow('frame1', self.color_image1)
                cv2.imshow('frame2', self.color_image2)
            
            except BaseException:
                pass
            



            if cv2.waitKey(1) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break

    def callback1(self, sub1, sub2, sub3, sub4):
        assert isinstance(sub1, Image)
        self.color_image1 = self.bridge.imgmsg_to_cv2(sub1, "bgr8")

        assert isinstance(sub2, Image)
        self.depth_image1 = self.bridge.imgmsg_to_cv2(sub2, "16UC1")

        assert isinstance(sub3, Image)
        self.color_image2 = self.bridge.imgmsg_to_cv2(sub3, "bgr8")

        assert isinstance(sub4, Image)
        self.depth_image2 = self.bridge.imgmsg_to_cv2(sub4, "16UC1")
    



    # def callback2(self, sub1_message, sub2_message):
    #     assert isinstance(sub1_message, CameraInfo)
    #     self.color_camera_info = sub1_message
    #     self.color_intrinsics = np.asarray(self.color_camera_info.K).reshape(3, 3)

    #     assert isinstance(sub2_message, CameraInfo)
    #     self.depth_camera_info = sub2_message
    #     self.depth_intrinsics = np.asarray(self.depth_camera_info.K).reshape(3, 3)

    # def capture_event(self, event, x, y, flags, params):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         self.mouse_click_position = pixel_to_point(self.color_intrinsics, [int(x), int(y)], self.depth_image)
    #         print('2D: (%d, %d)' % (x, y), '  ', end='')
    #         print('3D: (%2.5f, %2.5f, %2.5f)' % (self.mouse_click_position[0], self.mouse_click_position[1], self.mouse_click_position[2]))

    #         dataTrans = self.mouse_click_position.reshape(1, -1).squeeze(axis=0)
    #         self.pub_mouse_click_position.publish(dataTrans)


if __name__ == '__main__':
    rospy.init_node('RealSenseRosSet', anonymous=True)
    ip = RealSenseRosSet()
    ip.run()
