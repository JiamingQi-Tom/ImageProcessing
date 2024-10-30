#! /usr/bin/env python3
import sys
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
from General_method2 import *
from ConvertPointcloud import *
from plugins.msg import msg_int, msg_float
from copy import deepcopy
from imgpro_roslaunch import *
from cameraClass import LowPassFilter


class ImageProcessing(RealSenseRosSet):
    def __init__(self):
        super(ImageProcessing, self).__init__()
        self.publish_rate = 30.0
        self.rate = rospy.Rate(self.publish_rate)

        self.pub1 = rospy.Publisher("/marker_2D", msg_int, queue_size=10)
        self.pub2 = rospy.Publisher("/marker_3D", PointCloud2, queue_size=10)

        self.fixedNum = 50

        self.Ts = np.loadtxt('/home/tomqi/Documents/exps_ws/src/plugins/script/CameraCalibration/tf_base_to_camera2.txt')

        self.cropsize = np.array([[322, 121], [890, 654]])

        self.marker_2D = np.zeros((10, 2), dtype=np.uint8)
        self.marker_3D = np.zeros((10, 3), dtype=np.float)
        self.marker_3D_ = np.zeros((10, 3), dtype=np.float)

        self._markerfilterFlag = False

        rospy.sleep(1)

    def run(self):
        while not rospy.is_shutdown():
            frame1 = deepcopy(self.color_image)

            # ----------------------------------------------- RIM -----------------------------------------------#
            try:
                colors = [[13, 129, 175]]
                centers = find_colorpoints(frame1, colors, cropsize=self.cropsize)
                self.marker_2D = centers[0, :]

                dataTrans = self.marker_2D.reshape(1, -1).squeeze(axis=0)
                self.pub1.publish(dataTrans)

                self.marker_3D = Pixel_to_Point(self.color_intrinsics, self.marker_2D, self.depth_image)
                self.marker_3D = transform_shape(self.marker_3D.reshape(1, -1), self.Ts)

                dataTrans = xyzrgb2pointcloud2(self.marker_3D, [0, 255, 0], 'camera_link')
                self.pub2.publish(dataTrans)

            except BaseException:
                # dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)
                # self.pub1.publish(dataTrans)

                # dataTrans = xyzrgb2pointcloud2(np.array([0, 0, 0]), [0, 255, 0], 'camera_link')
                # self.pub2.publish(dataTrans)
                pass

            # ----------------------------------------------- SHAPE AND MARKER PLOT -----------------------------------------------#
            try:
                cv2.circle(frame1, (self.marker_2D[0], self.marker_2D[1]), 4, (0, 0, 255), -1)

                # for i in range(np.size(self.rim_2D, axis=0)):
                #     cv2.circle(frame1, (self.rim_2D[i, 0], self.rim_2D[i, 1]), 2, (0, 200, 10), -1)
                # for i in range(self.fixedNum):
                #     cv2.circle(frame1, (self.contour_2D[i, 0], self.contour_2D[i, 1]), 2, (0, 200, 10), -1)
                # for i in range(np.size(self.marker_2D, axis=0)):
                #     cv2.circle(frame1, (self.marker_2D[i, 0], self.marker_2D[i, 1]), 2, (255, 0, 0), -1)
                #     cv2.putText(frame1, str(i), (self.marker_2D[i, 0], self.marker_2D[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_4)

            except BaseException:
                pass

            cv2.rectangle(frame1, self.cropsize[0, :], self.cropsize[1, :], (0, 255, 0), 2)
            frame1 = frame1[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.imshow('frame1', frame1)

            if cv2.waitKey(1) * 0xff == ord('q'):
                cv2.destroyAllWindows()
                break

            # self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('imgpro_fabric', anonymous=True)
    ip = ImageProcessing()
    ip.run()
