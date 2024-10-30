#! /usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from method import *
from experiment4.msg import msg_int, msg_float
from math import *
#
#
# class ImageProcessing:
#     def __init__(self):
#         self._to = time.time()
#         # receive origin image from camera
#         self.image_sub = rospy.Subscriber("/image_view/output", Image, self.callback)
#
#         # publish/receive marker point
#         self.image_pub1 = rospy.Publisher("/left_right_marker_yellow", mymsg, queue_size=1)
#
#         # publish/receive centerline
#         self.image_pub2 = rospy.Publisher("/centerline", mymsg, queue_size=1)
#         self.image_sub2 = rospy.Subscriber("/reconstruct_centerline", mymsg, self.callback2)
#         self.reconstruct_centerline = np.zeros((1000, 2)).astype(int)
#
#         # self.image_pub1 = rospy.Publisher("/marker_point", mymsg, queue_size=1)
#         # self.image_sub1 = rospy.Subscriber("/reconstruct_marker_point", mymsg, self.callback1)
#         # self.reconstruct_marker_point = np.zeros((1, 2)).astype(int)
#
#         # self.image_pub3 = rospy.Publisher("/center_line_ordered", mymsg, queue_size=1)
#         # self.image_pub4 = rospy.Publisher("/center_line_minisom", mymsg, queue_size=1)
#         # self.image_pub5 = rospy.Publisher("/marker_center", mymsg, queue_size=1)
#         # self.image_pub6 = rospy.Publisher("/marker_center_distance_angle", mymsg, queue_size=1)
#         # self.image_pub7 = rospy.Publisher("/marker_center_distance", mymsg, queue_size=1)
#         # self.image_pub8 = rospy.Publisher("/left_right_marker", mymsg, queue_size=1)
#         # self.image_pub9 = rospy.Publisher("/left_right_marker_yellow", mymsg, queue_size=1)
#         # self.image_pub10 = rospy.Publisher("/yellow_ball", mymsg, queue_size=1)
#
#         self.bridge = CvBridge()
#
#     def callback(self, data):
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#         except CvBridgeError as e:
#             print e
#
#         # video capture
#         origin = deepcopy(cv_image)
#         frame1 = deepcopy(cv_image)
#         frame2 = deepcopy(cv_image)
#
#         # Find different markers
#         try:
#             left_red, right_red = find_double_red(origin)
#             left_yellow, right_yellow = find_double_yellow(origin, left_red, right_red)
#         except BaseException:
#             pass
#
#         cv2.line(frame1, (319, 0), (319, 639), (0, 255, 0), 1)
#
#         # Find red/yellow erea
#         try:
#             red_area = find_red_area(origin)
#             yellow_area = find_yellow_area(origin)
#         except BaseException:
#             pass
#
#         # Find the centerline of the black rod
#         try:
#             centerline = find_black_rod_centerline(origin, red_area, yellow_area)
#         except BaseException:
#             pass
#
#         try:
#             cv2.circle(frame1, (left_red[0], left_red[1]), 3, (0, 255, 0), -1)
#             cv2.circle(frame1, (right_red[0], right_red[1]), 3, (0, 255, 0), -1)
#             cv2.circle(frame1, (left_yellow[0], left_yellow[1]), 3, (0, 255, 0), -1)
#             cv2.circle(frame1, (right_yellow[0], right_yellow[1]), 3, (0, 255, 0), -1)
#
#             cv2.putText(frame1, '1', (left_red[0], left_red[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
#             cv2.putText(frame1, '2', (left_yellow[0], left_yellow[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
#             cv2.putText(frame1, '3', (right_red[0], right_red[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
#             cv2.putText(frame1, '4', (right_yellow[0], right_yellow[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
#
#             for i in range(np.size(centerline, 0)):
#                 cv2.circle(frame2, (centerline[i, 0], centerline[i, 1]), 1, (0, 255, 0), -1)
#
#             # for i in range(np.size(centerline_ordered_fixed, 0)):
#             #     cv2.circle(frame3, (centerline_ordered_fixed[i][0], centerline_ordered_fixed[i][1]), 1, (0, 255, 0), -1)
#
#             for i in range(np.size(self.reconstruct_centerline, 0)):
#                 cv2.circle(frame2, (self.reconstruct_centerline[i, 0], self.reconstruct_centerline[i, 1]), 1, (0, 0, 255), -1)
#
#         except BaseException:
#             pass
#
#         try:
#             dataTrans = np.asarray([left_red, left_yellow, right_red, right_yellow]).reshape(1, -1).squeeze(axis=0)
#             self.image_pub1.publish(dataTrans)
#         except BaseException:
#             pass
#
#         try:
#             dataTrans = np.asarray(centerline).reshape(1, -1).squeeze(axis=0)
#             self.image_pub2.publish(dataTrans)
#         except BaseException:
#             pass
#
#         cv2.imshow('frame1', frame1)
#         cv2.imshow('frame2', frame2)
#         cv2.waitKey(1)
#
#     def callback2(self, msg):
#         temp = np.asarray(msg.marker_centerline).astype(int).reshape(-1, 2)
#         self.reconstruct_centerline = temp
#
#
# if __name__ == '__main__':
#     rospy.init_node('image_processing', anonymous=True)
#     ip = ImageProcessing()
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         cv2.destroyAllWindows()
