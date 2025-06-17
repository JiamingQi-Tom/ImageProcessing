#! /usr/bin/env python3
import sys
import cv2

sys.path.append('/home/tomqi/Documents/exps_ws/src/plugin/script')
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugin/script/image_processing')
import rospy
import pyrealsense2 as rs
from General_method1 import *
from General_method2 import *
from ConvertPointcloud import *
from plugin.msg import msg_int, msg_float
from time import time
from cameraClass import *
from imgpro_exp11 import Obtain3DShape


class ImageProcessing(RealSenseD455Set):
    def __init__(self):
        super(ImageProcessing, self).__init__(displaysize='small')
        self.publish_rate = 30

        self.pub1 = rospy.Publisher("/real_marker_2D", msg_int, queue_size=2)
        self.pub2 = rospy.Publisher("/real_marker_3D", PointCloud2, queue_size=2)

        self.pub3 = rospy.Publisher("/shape_2D", msg_int, queue_size=2)
        self.pub4 = rospy.Publisher("/shape_3D", PointCloud2, queue_size=2)

        self._calibratedFlag = True
        self._markerfilterFlag = False
        self._shapefilterFlag = False

        if self._calibratedFlag:
            # self.Ts = np.loadtxt('/home/tomqi/Documents/exps_ws/src/plugin/srcipt/ImageProcessing/tf_base_to_camera.txt')
            self.Ts = np.loadtxt('/home/tomqi/Documents/exps_ws/src/plugin/script/ImageProcessing/tf_base_to_camera.txt')
        else:
            self.Ts = np.eye(4)

        self.fixedNum = 16
        self.cropsize = np.array([[50, 30], [540, 400]])

    def run(self):
        rate = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown():
            tic = time()
            color_image, depth_image, aligned_color_frame, aligned_depth_frame = self.get_aligned_images()
            frame1 = deepcopy(color_image)
            frame2 = deepcopy(color_image)
            # frame3 = deepcopy(color_image)

            # Marker extraction
            try:
                # colors = [[38, 71, 50], [12, 92, 155], [152, 219, 85]]
                colors = [[32, 114, 57], [10, 156, 123], [155, 112, 50]]
                centers = find_colorpoints(frame1, colors, cropsize=self.cropsize)
                green, yellow, red = centers[0, :], centers[1, :], centers[2, :]

                real_marker_2D = np.vstack((green, red, yellow))
                # real_marker_3D = zeros((size(real_marker_2D, 0), 3), dtype=float)
                real_marker_3D = Obtain3DShape(real_marker_2D, aligned_depth_frame, self.depth_intrin)[0]

                # for i in range(size(real_marker_2D, 0)):
                #     depth = aligned_depth_frame.get_distance(real_marker_2D[i, 0], real_marker_2D[i, 1])
                #     real_marker_3D[i, :] = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [real_marker_2D[i, 0], real_marker_2D[i, 1]], depth)

                # if not self._markerfilterFlag:
                #     real_marker_3D_ = real_marker_3D
                #     self._markerfilterFlag = True
                #     P0 = eye(size(real_marker_3D, 0) * size(real_marker_3D, 1)) * 1e-3
                #     vars = eye(size(real_marker_3D, 0) * size(real_marker_3D, 1)) * 3e-4
                #     varr = eye(size(real_marker_3D, 0) * size(real_marker_3D, 1)) * 1e-2
                # else:
                #     real_marker_3D, P = LKFShapeFilter(real_marker_3D_, real_marker_3D, P0, var_s=vars, var_r=varr)
                #     real_marker_3D_ = real_marker_3D
                #     P0 = P

                    # real_marker_3D = LowPassFIlter(real_marker_3D_, real_marker_3D, 0.4)
                    # real_marker_3D_ = real_marker_3D

                dataTrans = np.array(real_marker_2D).reshape(1, -1).squeeze(axis=0)
                self.pub1.publish(dataTrans)

                real_marker_3D = transform_shape(real_marker_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(real_marker_3D, [0, 255, 0], frame_id='world')
                self.pub2.publish(dataTrans)

            except BaseException:
                pass

            # Shape extraction
            try:
                # yellow = np.array([70, 400])
                # shape_2D = find_blue_centerline(frame1, color=[83, 131, 34], fixedNum=16, base=yellow, cropsize=None)[0]
                shape_2D = find_blue_centerline(frame1, color=[83, 131, 34], fixedNum=self.fixedNum, base=yellow, cropsize=None)[0]
                shape_3D = Obtain3DShape(shape_2D, aligned_depth_frame, self.depth_intrin)[0]

                if not self._shapefilterFlag:
                    shape_3D_ = shape_3D
                    self._shapefilterFlag = True
                else:
                    shape_3D = LowPassFilter(shape_3D_, shape_3D, 0.4)
                    shape_3D_ = shape_3D

                dataTrans = shape_2D.reshape(1, -1).squeeze(axis=0)
                self.pub3.publish(dataTrans)

                shape_3D = transform_shape(shape_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(shape_3D, [0, 255, 0], 'world')
                self.pub4.publish(dataTrans)

            except BaseException:
                pass

            try:
                for i in range(np.size(real_marker_2D, 0)):
                    cv2.circle(frame1, (real_marker_2D[i, 0], real_marker_2D[i, 1]), 3, (255, 10, 10), -1)
                    cv2.putText(frame1, '%d' % i, (real_marker_2D[i, 0] - 8, real_marker_2D[i, 1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
            except BaseException:
                pass

            try:
                for i in range(self.fixedNum):
                    cv2.circle(frame2, (shape_2D[i, 0], shape_2D[i, 1]), 3, (0, 0, 255), -1)

                cv2.putText(frame2, '0', (shape_2D[0, 0] - 2, shape_2D[0, 1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)
                cv2.putText(frame2, '15', (shape_2D[15, 0] - 2, shape_2D[15, 1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)

            except BaseException:
                pass

            try:
                cv2.putText(frame1, '%d Hz' % (1.0 / (time() - tic)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_4)
            except BaseException:
                pass

            cv2.rectangle(frame1, self.cropsize[0, :], self.cropsize[1, :], (0, 255, 0), 2)

            cv2.imshow('frame1', frame1)
            cv2.imshow('frame2', frame2)

            # cv2.imshow('frame3', frame3)
            # cv2.imshow('frame4', frame4)
            # all_four_in_one = vstack([hstack([frame1, frame2]), hstack([frame3, frame4])])
            # cv2.imshow('all_four_in_one', all_four_in_one)

            if cv2.waitKey(1) * 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('image_processing', anonymous=True)
    ip = ImageProcessing()
    ip.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
