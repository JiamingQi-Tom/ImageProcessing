#! /usr/bin/env python3
import cv2
import sys
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script/image_processing')
from ConvertPointcloud import *
from plugins.msg import msg_int, msg_float
from copy import deepcopy
from ClusterAlgorithms import *
from imgpro_general import find_area, points_sorting


def find_white_centerline(orgin, threshold=80, fixedNum=30, base=np.array([247, 59]), cropsize=None):
    gray = cv2.cvtColor(orgin, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    if cropsize is not None:
        closing = deepcopy(closing[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(closing, kernel, iterations=2)

    # cv2.imshow('erode', erode)

    contours, hierarchy, = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]
    contour = contours[int(np.argmax(np.array(contoursArea)))]

    contour = np.squeeze(contour, axis=1)

    newbinary = cv2.fillPoly(np.zeros_like(closing, dtype=np.uint8), [contour], (255))

    thinned = cv2.ximgproc.thinning(newbinary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    centerline = np.hstack((np.where(thinned == 255)[1].reshape((-1, 1)), np.where(thinned == 255)[0].reshape((-1, 1))))

    centerline = ClusteringExtractionPoints('FPS', centerline, fixedNum=fixedNum)

    if cropsize is not None:
        centerline = centerline + np.array([cropsize[0, 0], cropsize[0, 1]])

    # base = array([base[0], base[1]])
    distance = []
    N = np.size(centerline, 0)
    ordered = np.zeros((N, 2), dtype=int)
    for i in range(N):
        for j in range(np.size(centerline, 0)):
            distance.append(np.linalg.norm(base - centerline[j, :], ord=2))
        index = np.argmin(np.array(distance))
        ordered[i, :] = centerline[index, :]
        centerline = np.delete(centerline, index, axis=0)
        distance = []
        base = ordered[i, :]

    return ordered, closing


def find_blue_centerline(origin, color, fixedNum=30, base=np.array([247, 59]), cropsize=None):
    closing = find_area(origin, threshold=color, hsv_flag=True)[1]

    if cropsize is not None:
        closing = deepcopy(closing[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(closing, kernel, iterations=2)

    contours, hierarchy, = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]
    contour = contours[int(np.argmax(np.array(contoursArea)))]

    contour = np.squeeze(contour, axis=1)

    newbinary = cv2.fillPoly(np.zeros_like(closing, dtype=np.uint8), [contour], (255))

    thinned = cv2.ximgproc.thinning(newbinary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    centerline = np.hstack((np.where(thinned == 255)[1].reshape((-1, 1)), np.where(thinned == 255)[0].reshape((-1, 1))))

    # cv2.imshow('thinned', thinned)

    centerline = ClusteringExtractionPoints('FPS', centerline, fixedNum=fixedNum)

    if cropsize is not None:
        centerline = centerline + np.array([cropsize[0, 0], cropsize[0, 1]])

    centerline = points_sorting(centerline, base=base)

    return centerline, closing


def find_contour_middle2(origin, threshold, fixedNum=20, base=np.array([245, 61]), cropsize=None):
    closing = find_area(origin, threshold, hsv_flag=False)[1]

    if cropsize is not None:
        closing = deepcopy(closing[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(closing, kernel, iterations=1)

    contours, hierarchy, = cv2.findContours(erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]

    top_k_idx = np.array(contoursArea).argsort()[::-1][0:2]

    idx_1st = top_k_idx[0]
    idx_2rd = top_k_idx[1]

    contour_1st = contours[idx_1st]
    contour_1st = np.squeeze(contour_1st, axis=1)

    contour_2rd = contours[idx_2rd]
    contour_2rd = np.squeeze(contour_2rd, axis=1)

    newbinary = cv2.fillPoly(np.zeros_like(closing, dtype=np.uint8), [contour_1st], (255))
    newbinary = cv2.fillPoly(newbinary, [contour_2rd], (0))

    # cv2.imshow('newbinary', newbinary)

    # Extract centerline
    thinned = cv2.ximgproc.thinning(newbinary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    centerline = np.hstack((np.where(thinned == 255)[1].reshape((-1, 1)), np.where(thinned == 255)[0].reshape((-1, 1))))

    # cv2.imshow('thinned', thinned)

    if cropsize is not None:
        centerline = centerline + cropsize[0, :]
        center = np.mean(centerline, axis=0).astype(np.float) + cropsize[0, :]
    else:
        center = np.mean(centerline, axis=0).astype(np.float)

    # Down-sampling
    centerline = ClusteringExtractionPoints('FPS', centerline, fixedNum=fixedNum)

    # Sorting
    centerline = points_sorting(centerline, base=base)

    return centerline, center


def find_contour(origin, color, fixedNum=30, cropsize=None):
    closing = find_area(origin, threshold=color, hsv_flag=True)[1]

    if cropsize is not None:
        closing = deepcopy(closing[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(closing, kernel, iterations=2)

    contours, hierarchy, = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]
    contour = contours[int(np.argmax(np.array(contoursArea)))]

    contour = np.squeeze(contour, axis=1)

    contour = ClusteringExtractionPoints('FPS', contour, fixedNum=fixedNum)

    contour = np.asarray(contour, dtype=np.int)

    if cropsize is not None:
        contour = contour + np.array([cropsize[0, 0], cropsize[0, 1]])

    center = np.mean(contour, axis=0)

    contour = points_sorting(contour, base=center)

    return contour, center


class ImageProcessing(RealSenseD455Set):
    def __init__(self):
        super(ImageProcessing, self).__init__(displaysize='big')
        self.publish_rate = 30

        self.pub1 = rospy.Publisher("/shape_2D", msg_int, queue_size=2)
        self.pub2 = rospy.Publisher("/shape_3D", PointCloud2, queue_size=2)

        self.pub3 = rospy.Publisher("/feature", msg_float, queue_size=2)

        self.fixedNum = 20

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

            shape_case = 2

            try:
                if shape_case == 1:
                    shape_2D = find_white_centerline(frame1, threshold=45, fixedNum=self.fixedNum, base=[589, 642], cropsize=self.cropsize)[0]

                elif shape_case == 2:
                    # shape_2D = find_blue_centerline(frame1, color=[98, 81, 168], fixedNum=self.fixedNum, base=[589, 642], cropsize=self.cropsize)[0]
                    shape_2D = find_blue_centerline(frame1, color=[97, 110, 64], fixedNum=self.fixedNum, base=[313, 589] + self.cropsize[0, :], cropsize=self.cropsize)[0]

                elif shape_case == 3:
                    shape_2D, center = find_contour_middle2(frame1, threshold=55, fixedNum=self.fixedNum, base=np.array([293, 595] + self.cropsize[0, :]), cropsize=self.cropsize)

                elif shape_case == 4:
                    shape_2D, center = find_contour(frame1, color=[14, 64, 98], fixedNum=self.fixedNum, cropsize=self.cropsize)

                # if not self.shapefilter_flag:
                #     shape_2D_ = shape_2D
                #     self.shapefilter_flag = True
                # else:
                #     shape_2D = np.asarray(LowPassFilter(shape_2D_, shape_2D, 1.0), dtype=np.int)
                #     shape_2D_ = shape_2D

                shape2D = np.loadtxt('/home/tomqi/Documents/exps_ws/src/exp2/script/5.External Disturbance/data.txt', delimiter=',')[:, 1:41]
                desired = np.asarray(shape2D[0, :].reshape(-1, 2), dtype=np.int)
                desired = np.vstack((desired, desired[0, :]))
                for i in range(19):
                    cv2.line(frame2, (desired[i, 0], desired[i, 1]), (desired[i + 1, 0], desired[i + 1, 1]), (0, 0, 255), 3)

                if shape_case == 3 or shape_case == 4:
                    cv2.circle(frame1, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)

                # for i in range(self.fixedNum - 1):
                #     cv2.line(frame1, (shape_2D[i, 0], shape_2D[i, 1]), (shape_2D[i + 1, 0], shape_2D[i + 1, 1]), (0, 0, 255), 3)

                # for i in range(self.fixedNum):
                #     cv2.circle(frame1, (shape_2D[i, 0], shape_2D[i, 1]), 3, (0, 0, 255), -1)
                #     cv2.putText(frame1, str(i), (shape_2D[i, 0], shape_2D[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)

                dataTrans = shape_2D.reshape(1, -1).squeeze(axis=0)
                self.pub1.publish(dataTrans)

                if shape_case == 1 or shape_case == 2:
                    dataTrans = np.array([0, 0], dtype=np.float).reshape(1, -1).squeeze(axis=0)

                else:
                    dataTrans = center.reshape(1, -1).squeeze(axis=0)

                self.pub3.publish(dataTrans)

            except BaseException:
                dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)

                self.pub1.publish(dataTrans)
                self.pub3.publish(dataTrans)

            try:
                shape_3D = Obtain3DShape(shape_2D, aligned_depth_frame, self.depth_intrin)[0]

                shape_3D = transform_shape(shape_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(shape_3D, [255, 0, 0], 'world')
                self.pub2.publish(dataTrans)

            except BaseException:
                dataTrans = xyzrgb2pointcloud2(np.array([0, 0, 0]), [255, 0, 0], 'world')

                self.pub2.publish(dataTrans)

            # cv2.rectangle(frame1, self.cropsize[0, :], self.cropsize[1, :], (0, 255, 0), 2)

            frame1 = frame1[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.imshow('frame1', frame1)
            frame2 = frame2[self.cropsize[0, 1]:self.cropsize[1, 1], self.cropsize[0, 0]:self.cropsize[1, 0]]
            cv2.imshow('frame2', frame2)

            if cv2.waitKey(1) * 0xff == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    rospy.init_node('imgpro_fabric', anonymous=True)
    ip = ImageProcessing()
    ip.run()

