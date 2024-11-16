#! /usr/bin/env python3
import sys
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
sys.path.append('/home/tomqi/Documents/exps_ws/src/exp_m7/script')
import cv2
from ConvertPointcloud import *
from copy import deepcopy
from cameraClass_ROS import *
from imgpro_general import find_area
from plugins.msg import float_array

def find_bag_rim(origin, color, fixedNum=30, base=[247, 59], cropsize=None, clustering=True):
    closing = find_area(origin, threshold=color, hsv_flag=False)[1]

    if cropsize is not None:
        closing = deepcopy(closing[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.erode(closing, kernel, iterations=8)

    try:
        contours, hierarchy, = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]

        newbinary = np.zeros_like(closing)

        top_k_idx = np.array(contoursArea).argsort()[::-1][0:1]

        for i in range(np.size(top_k_idx)):
            idx = top_k_idx[i]
            contour = np.squeeze(contours[idx], axis=1)
            newbinary = cv2.fillPoly(newbinary, [contour], (255))

        [x, y] = np.where(newbinary == 255)
        points = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))).astype(np.int)
        if cropsize is not None:
            points = points + np.array([cropsize[0, 1], cropsize[0, 0]])

        if clustering:
            points = ClusteringExtractionPoints('KMS', points, fixedNum=fixedNum).astype(np.int)

        points[:, [0, -1]] = points[:, [-1, 0]]

    except:
        pass

    return points


class ImageProcessing(RealSenseRosSet):
    def __init__(self):
        super(ImageProcessing, self).__init__()
        # self.publish_rate = 30.0
        # self.rate = rospy.Rate(self.publish_rate)

        self.bag_2D = np.zeros((10, 2), dtype=np.uint8)
        self.pub1 = rospy.Publisher("/bag_2D", float_array, queue_size=10)

        self.bag_3D = np.zeros((10, 3), dtype=np.float64)
        self.pub2 = rospy.Publisher("/bag_3D", PointCloud2, queue_size=10)

        self.query_point_2D = np.zeros(2, dtype=np.uint8)
        self.pub3 = rospy.Publisher("/query_point_2D", float_array, queue_size=10)

        self.query_point_3D = np.zeros(3, dtype=np.float64)
        self.pub4 = rospy.Publisher("/query_point_3D", PointCloud2, queue_size=10)

        self.neighbours_point_2D = np.zeros((10, 2), dtype=np.uint8)
        self.pub5 = rospy.Publisher("/neighbours_point_2D", float_array, queue_size=10)

        self.neighbours_point_3D = np.zeros((10, 3), dtype=np.float64)
        self.pub6 = rospy.Publisher("/neighbours_point_3D", PointCloud2, queue_size=10)

        self.random_point_2D = np.zeros(2, dtype=np.uint8)
        self.random_point_3D = np.zeros(3, dtype=np.float64)

        self.fixedNum = 20
        self.cropsize = np.array([[443, 211], [930, 556]])
        # self.Ts = np.loadtxt('/home/tomqi/Documents/exps_ws/src/plugins/script/CameraCalibration/dataset/tf_right_base_to_camera_1.txt')

        rospy.sleep(1)

    def run(self):
        while not rospy.is_shutdown():
            frame1 = deepcopy(self.color_image)
            # frame2 = deepcopy(self.color_image)

            try:
                # ------------------------------------ bag points ------------------------------------ #
                self.bag_2D = find_bag_rim(frame1, 50, fixedNum=self.fixedNum, base=[247, 59], cropsize=self.cropsize, clustering=False)
                dataTrans = self.bag_2D.reshape(1, -1).squeeze(axis=0)
                self.pub1.publish(dataTrans)

                # for i in range(np.size(self.bag_2D, axis=0)):
                #     cv2.circle(frame1, (self.bag_2D[i, 0], self.bag_2D[i, 1]), 1, (40, 40, 150), -1)

                self.bag_3D = pixel_to_point(self.color_intrinsics, self.bag_2D, self.depth_image)
                dataTrans = xyzrgb2pointcloud2(self.bag_3D, self.bag_2D, self.color_image, False, [0, 255, 0], 'camera_link')
                self.pub2.publish(dataTrans)

                # ------------------------------------ query points ------------------------------------ #
                # self.query_point_2D = np.array([638, 352], dtype=np.int)
                # dataTrans = self.query_point_2D.reshape(1, -1).squeeze(axis=0)
                # self.pub3.publish(dataTrans)
                #
                # cv2.circle(frame1, (self.query_point_2D[0], self.query_point_2D[1]), 3, (0, 200, 10), -1)
                #
                # self.query_point_3D = pixel_to_point(self.color_intrinsics, self.query_point_2D, self.depth_image, fixed_depth=False)
                # dataTrans = xyzrgb2pointcloud2(self.query_point_3D, self.query_point_2D, self.color_image, True, [0, 255, 0], 'camera_link')
                # self.pub4.publish(dataTrans)

                # ------------------------------------ neighbours points ------------------------------------ #
                # self.neighbours_point_2D = KNN_BallTree(self.bag_2D, query_point=self.query_point_2D, K=400)
                # dataTrans = self.neighbours_point_2D.reshape(1, -1).squeeze(axis=0)
                # self.pub5.publish(dataTrans)
                #
                # for i in range(np.size(self.neighbours_point_2D, axis=0)):
                #     cv2.circle(frame1, (self.neighbours_point_2D[i, 0], self.neighbours_point_2D[i, 1]), 1, (40, 40, 150), -1)
                #
                # self.neighbours_point_3D = pixel_to_point(self.color_intrinsics, self.neighbours_point_2D, self.depth_image, fixed_depth=False)
                # # self.neighbours_point_3D = transform_shape(self.neighbours_point_3D, self.Ts)
                # dataTrans = xyzrgb2pointcloud2(self.neighbours_point_3D, self.neighbours_point_2D, self.color_image, False, [0, 255, 0], 'camera_link')
                # self.pub6.publish(dataTrans)

                # ------------------------------------ random points ------------------------------------ #
                # random_point_2D = self.bag_2D[np.random.randint(0, np.size(self.bag_2D, axis=0) - 1), :]
                # self.bag_3D = pixel_to_point(self.color_intrinsics, random_point_2D, self.depth_image, fixed_depth=False).reshape(1, -1)
                # for i in range(np.size(near_points, axis=0)):
                #     cv2.circle(frame1, (near_points[i, 0], near_points[i, 1]), 1, (40, 40, 150), -1)

            except BaseException:
                pass

            # ----------------------------------------------- VISUALIZING -----------------------------------------------#
            # frame1 = frame1[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.rectangle(frame1, self.cropsize[0, :], self.cropsize[1, :], (0, 255, 0), 2)
            cv2.imshow('frame1', frame1)

            # cv2.rectangle(frame2, self.cropsize[0, :], self.cropsize[1, :], (0, 255, 0), 2)
            # cv2.imshow('frame2', frame2)

            # if cv2.waitKey(1) * 0xff == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
            # elif cv2.waitKey(1) & 0xFF == ord('s'):
            #     np.savetxt('neighbours_point_3D.txt', self.neighbours_point_3D, delimiter=',')
            cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('imgpro_fabric', anonymous=True)
    ip = ImageProcessing()
    ip.run()
