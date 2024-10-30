#! /usr/bin/env python3
# import cv2
import sys
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script')
sys.path.append('/home/tomqi/Documents/exps_ws/src/plugins/script/image_processing')
import message_filters
from imgpro_general import points_sorting
from std_msgs.msg import Int32
from ClusterAlgorithms import *
from cameraClass_ROS import *
from cameraClass_API import *


def HoughLineDetection(gray):
    # Step 1: Apply edge detection
    edges = cv2.Canny(gray, 60, 220, apertureSize=3)

    # Step 2: Apply Hough line transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)

    # Step 3: Extract the first index as the longest line
    rho = lines[0, 0, 0]
    theta = lines[0, 0, 1]
    a = np.cos(theta)
    b = np.sin(theta)

    # Step 4: Obtain the point-slope form
    slope = -a / b
    bias = rho / b
    paras1 = [slope, bias]

    # Step 5: Obtain the general form
    A = slope
    B = -1.0
    C = bias
    paras2 = [A, B, C]

    return paras1, paras2


def find_down_top_fabric(img, fixedNum, cropsize=None):
    # Step 1: pre-processing of img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    if cropsize is not None:
        thresh = deepcopy(thresh[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(closing, kernel, iterations=1)

    kernel = np.ones((9, 9), np.uint8)
    erode = cv2.erode(dilate, kernel, iterations=1)

    # cv2.imshow('erode', erode)

    # Step 2: Calculate 2D shape of top and down fabric
    contours, hierarchy, = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]

    top_k_idx = np.array(contoursArea).argsort()[::-1][0:2]

    down_fabric_shape_2D = np.squeeze(contours[top_k_idx[0]], axis=1) + np.array([cropsize[0, 0], cropsize[0, 1]])
    top_fabric_shape_2D = np.squeeze(contours[top_k_idx[1]], axis=1) + np.array([cropsize[0, 0], cropsize[0, 1]])

    f = FPS(down_fabric_shape_2D)
    down_fabric_shape_2D = f.comput_fps(fixedNum).astype(int)

    f = FPS(top_fabric_shape_2D)
    top_fabric_shape_2D = f.comput_fps(fixedNum).astype(int)

    # fabric_shape sorting
    down_fabric_shape_2D = points_sorting(down_fabric_shape_2D, deepcopy(down_fabric_shape_2D[0, :]))
    top_fabric_shape_2D = points_sorting(top_fabric_shape_2D, deepcopy(top_fabric_shape_2D[0, :]))

    down_fabric_center_2D = np.mean(down_fabric_shape_2D, axis=0).astype(int)
    top_fabric_center_2D = np.mean(top_fabric_shape_2D, axis=0).astype(int)

    down_fabric = [down_fabric_shape_2D, down_fabric_center_2D, contoursArea[top_k_idx[0]]]
    top_fabric = [top_fabric_shape_2D, top_fabric_center_2D, contoursArea[top_k_idx[1]]]

    # Step 3: Calculate 2D long sides of top and down fabric
    contours, hierarchy, = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]

    top_k_idx = np.array(contoursArea).argsort()[::-1][0:2]

    # Step 3-1: Calculate 2D long side of down fabric
    down_fabric_shape_2D_dilate = np.squeeze(contours[top_k_idx[0]], axis=1) + np.array([cropsize[0, 0], cropsize[0, 1]])
    down_fabric_edge = np.zeros_like(gray, dtype=np.uint8)
    down_fabric_edge[down_fabric_shape_2D_dilate[:, 1], down_fabric_shape_2D_dilate[:, 0]] = 255

    lineparas = HoughLineDetection(down_fabric_edge)[1]
    A, B, C = lineparas[0], lineparas[1], lineparas[2]

    distance = np.zeros(fixedNum, dtype=np.float)
    sidepoints = []
    for i in range(fixedNum):
        x = down_fabric_shape_2D[i, 0]
        y = down_fabric_shape_2D[i, 1]
        distance[i] = np.abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)

        if distance[i] < 8.0:
            sidepoints.append(np.array([x, y]))

    down_fabric.append(np.asarray(sidepoints))

    # Step 3-2: Calculate 2D long side of top fabric
    top_fabric_shape_2D_dilate = np.squeeze(contours[top_k_idx[1]], axis=1) + np.array([cropsize[0, 0], cropsize[0, 1]])
    top_fabric_edge = np.zeros_like(gray, dtype=np.uint8)
    top_fabric_edge[top_fabric_shape_2D_dilate[:, 1], top_fabric_shape_2D_dilate[:, 0]] = 255

    lineparas = HoughLineDetection(top_fabric_edge)[1]
    A, B, C = lineparas[0], lineparas[1], lineparas[2]

    distance = np.zeros(fixedNum, dtype=np.float)
    sidepoints = []
    for i in range(fixedNum):
        x = top_fabric_shape_2D[i, 0]
        y = top_fabric_shape_2D[i, 1]
        distance[i] = np.abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)

        if distance[i] < 8.0:
            sidepoints.append(np.array([x, y]))

    top_fabric.append(np.asarray(sidepoints))

    return down_fabric, top_fabric


def find_FabricFeature(img, cropsize=None, thrValue=160, findTop=True, top_contour=None, side='left'):
    # Step 1: pre-processing of img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if findTop:
        ret, thresh = cv2.threshold(gray, thrValue, 255, cv2.THRESH_BINARY)

        if cropsize is not None:
            thresh = deepcopy(thresh[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        kernel = np.ones((3, 3), np.uint8)
        erode = cv2.erode(closing, kernel, iterations=2)

        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(erode, kernel, iterations=4)

        # cv2.imshow('gray', gray)
        # cv2.imshow('dilate', dilate)

        contours, hierarchy, = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]

        if len(contours) == 0:
            return np.array([0, 0], dtype=np.int32), np.array([0, 0], dtype=np.int32).reshape(-1, 2)

        contour = contours[int(np.argmax(np.array(contoursArea)))]
        contour = np.squeeze(contour, axis=1) + np.array([cropsize[0, 0], cropsize[0, 1]])

        if side == 'left':
            distance = []
            for i in range(np.size(contour, axis=0)):
                distance.append(np.linalg.norm(contour[i, :], ord=2))

            top_corner_point = contour[np.argmin(np.asarray(distance)), :]
        elif side == 'right':
            distance = []
            for i in range(np.size(contour, axis=0)):
                distance.append(np.linalg.norm(contour[i, :] - np.array([639, 0]), ord=2))

            top_corner_point = contour[np.argmin(np.asarray(distance)), :]

        else:
            top_corner_point = np.array([0, 0])
            print('Side wrong!, please input again !')

        return top_corner_point, contour

    else:
        ret, thresh = cv2.threshold(gray, thrValue, 255, cv2.THRESH_BINARY)

        if cropsize is not None:
            thresh = deepcopy(thresh[cropsize[0, 1]:cropsize[1, 1], cropsize[0, 0]:cropsize[1, 0]])

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        kernel = np.ones((3, 3), np.uint8)
        erode = cv2.erode(closing, kernel, iterations=2)

        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(erode, kernel, iterations=1)

        # cv2.imshow('gray', gray)
        # cv2.imshow('dilate', dilate)

        contours, hierarchy, = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]

        if len(contours) == 0:
            return np.array([0, 0], dtype=np.int32)

        contour = contours[int(np.argmax(np.array(contoursArea)))]
        contour = np.squeeze(contour, axis=1) + np.array([cropsize[0, 0], cropsize[0, 1]])

        newbinary = cv2.fillPoly(np.zeros_like(gray, dtype=np.uint8), [contour], (255))
        newbinary = cv2.fillPoly(newbinary, [top_contour], (0))

        contours, hierarchy, = cv2.findContours(newbinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contoursArea = [cv2.contourArea(contour) for idx, contour in enumerate(contours)]

        contour = contours[int(np.argmax(np.array(contoursArea)))]

        if side == 'left':
            distance = []
            for i in range(np.size(contour, axis=0)):
                distance.append(np.linalg.norm(contour[i, :], ord=2))

            dowm_corner_point = contour[np.argmin(np.asarray(distance)), :]
        elif side == 'right':
            distance = []
            for i in range(np.size(contour, axis=0)):
                distance.append(np.linalg.norm(contour[i, :] - np.array([639, 0]), ord=2))

            dowm_corner_point = contour[np.argmin(np.asarray(distance)), :]

        else:
            dowm_corner_point = np.array([0, 0])
            print('Side wrong!, please input again !')

        return dowm_corner_point.squeeze(axis=0)


class ImageProcessing(RealSenseRosSet):
    def __init__(self):
        super(ImageProcessing, self).__init__()
        self.publish_rate = 30.0
        self.rate = rospy.Rate(self.publish_rate)

        self.pub1 = rospy.Publisher("/down_fabric_shape_2D", msg_int, queue_size=2)
        self.pub2 = rospy.Publisher("/top_fabric_shape_2D", msg_int, queue_size=2)

        self.pub3 = rospy.Publisher("/down_fabric_shape_3D", PointCloud2, queue_size=2)
        self.pub4 = rospy.Publisher("/top_fabric_shape_3D", PointCloud2, queue_size=2)

        self.pub5 = rospy.Publisher("/down_fabric_sidepoints_2D", msg_int, queue_size=2)
        self.pub6 = rospy.Publisher("/top_fabric_sidepoints_2D", msg_int, queue_size=2)

        self.pub7 = rospy.Publisher("/down_fabric_sidepoints_3D", PointCloud2, queue_size=2)
        self.pub8 = rospy.Publisher("/top_fabric_sidepoints_3D", PointCloud2, queue_size=2)

        self.pub10 = rospy.Publisher("/left_top_fabric_corner", msg_int, queue_size=2)
        self.pub11 = rospy.Publisher("/left_down_fabric_corner", msg_int, queue_size=2)
        self.pub12 = rospy.Publisher("/right_top_fabric_corner", msg_int, queue_size=2)
        self.pub13 = rospy.Publisher("/right_down_fabric_corner", msg_int, queue_size=2)

        self.display_state = 4
        self.sub1 = message_filters.Subscriber('/display_state', Int32, queue_size=1)

        self.sync = message_filters.ApproximateTimeSynchronizer([self.sub1], 10, 0.1, allow_headerless=True)
        self.sync.registerCallback(self.callback)

        self.fixedNum = 40

        self.left_top_fabric_corner = np.array([0, 0], dtype=np.int32)
        self.left_top_fabric_contour = np.array([0, 0], dtype=np.int32).reshape(-1, 2)
        self.left_down_fabric_corner = np.array([0, 0], dtype=np.int32)

        self.right_top_fabric_corner = np.array([0, 0], dtype=np.int32)
        self.right_top_fabric_contour = np.array([0, 0], dtype=np.int32).reshape(-1, 2)
        self.right_down_fabric_corner = np.array([0, 0], dtype=np.int32)

        self.top_down_fabric_cropsize = np.array([[400, 110], [850, 470]])

        self.left_top_fabric_cropsize = np.array([[145, 260], [400, 439]])
        self.right_top_fabric_cropsize = np.array([[274, 270], [519, 433]])

        self.left_down_fabric_cropsize = np.array([[100, 100], [540, 430]])
        self.right_down_fabric_cropsize = np.array([[100, 100], [540, 430]])

        self.camera_calibration_flag = True
        self.markerfilter_flag = False
        self.shapefilter_flag = False

        if self.camera_calibration_flag:
            self.Ts = np.loadtxt(ccpath + '/CameraCalibration/Python/dataset/tf_base_to_camera.txt')
        else:
            self.Ts = np.eye(4)

        self.D405_use_flag = True
        if self.D405_use_flag:
            self.D405 = RealSenseD405Set(displaysize='small')

    def run(self):
        while not rospy.is_shutdown():
            frame1 = deepcopy(self.color_image)
            frame2 = deepcopy(self.color_image)

            # 2D / 3D Fabric detection
            try:
                down_fabric, top_fabric = find_down_top_fabric(frame1, fixedNum=self.fixedNum, cropsize=self.top_down_fabric_cropsize)

                down_fabric_shape_2D = down_fabric[0]
                down_fabric_center_2D = down_fabric[1]
                down_fabric_sidepoints_2D = down_fabric[3]

                top_fabric_shape_2D = top_fabric[0]
                top_fabric_center_2D = top_fabric[1]
                top_fabric_sidepoints_2D = top_fabric[3]

                if self.display_state == 4:
                    for i in range(self.fixedNum):
                        cv2.circle(frame1, (down_fabric_shape_2D[i, 0], down_fabric_shape_2D[i, 1]), 3, (0, 0, 255), -1)
                        cv2.circle(frame1, (top_fabric_shape_2D[i, 0], top_fabric_shape_2D[i, 1]), 3, (0, 0, 255), -1)

                        # if np.mod(i, 3) == 0:
                #             cv2.putText(frame1, str(i), (down_fabric_shape_2D[i, 0], down_fabric_shape_2D[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_4)
                #             cv2.putText(frame1, str(i), (top_fabric_shape_2D[i, 0], top_fabric_shape_2D[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_4)
                #
                    for i in range(np.size(down_fabric_sidepoints_2D, axis=0)):
                        cv2.circle(frame1, (down_fabric_sidepoints_2D[i, 0], down_fabric_sidepoints_2D[i, 1]), 6, (0, 200, 10), -1)

                    for i in range(np.size(top_fabric_sidepoints_2D, axis=0)):
                        cv2.circle(frame1, (top_fabric_sidepoints_2D[i, 0], top_fabric_sidepoints_2D[i, 1]), 6, (0, 200, 10), -1)

                # cv2.putText(frame1, 'Down, area: %d' % down_fabric[2], down_fabric_center_2D, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_4)
                # cv2.putText(frame1, 'Top,  area: %d' % top_fabric[2], top_fabric_center_2D, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_4)

                dataTrans = down_fabric_shape_2D.reshape(1, -1).squeeze(axis=0)
                self.pub1.publish(dataTrans)

                dataTrans = top_fabric_shape_2D.reshape(1, -1).squeeze(axis=0)
                self.pub2.publish(dataTrans)

                dataTrans = down_fabric_sidepoints_2D.reshape(1, -1).squeeze(axis=0)
                self.pub5.publish(dataTrans)

                dataTrans = top_fabric_sidepoints_2D.reshape(1, -1).squeeze(axis=0)
                self.pub6.publish(dataTrans)

            except BaseException:
                pass

            try:
                down_fabric_shape_3D = pixel_to_point(self.color_intrinsics, down_fabric_shape_2D, self.depth_image)
                top_fabric_shape_3D = pixel_to_point(self.color_intrinsics, top_fabric_shape_2D, self.depth_image)
                down_fabric_sidepoints_3D = pixel_to_point(self.color_intrinsics, down_fabric_sidepoints_2D, self.depth_image)
                top_fabric_sidepoints_3D = pixel_to_point(self.color_intrinsics, top_fabric_sidepoints_2D, self.depth_image)

                down_fabric_shape_3D = transform_shape(down_fabric_shape_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(down_fabric_shape_3D, down_fabric_shape_2D, self.color_image, True, [217, 83, 25], 'world')
                self.pub3.publish(dataTrans)

                top_fabric_shape_3D = transform_shape(top_fabric_shape_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(top_fabric_shape_3D, top_fabric_shape_2D, self.color_image, True, [217, 83, 25], 'world')
                self.pub4.publish(dataTrans)

                down_fabric_sidepoints_3D = transform_shape(down_fabric_sidepoints_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(down_fabric_sidepoints_3D, down_fabric_sidepoints_2D, self.color_image, True, [119, 172, 48], 'world')
                self.pub7.publish(dataTrans)

                top_fabric_sidepoints_3D = transform_shape(top_fabric_sidepoints_3D, self.Ts)
                dataTrans = xyzrgb2pointcloud2(top_fabric_sidepoints_3D, top_fabric_sidepoints_2D, self.color_image, True, [119, 172, 48], 'world')
                self.pub8.publish(dataTrans)

            except BaseException:
                pass

            # cv2.rectangle(frame1, self.top_down_fabric_cropsize[0, :], self.top_down_fabric_cropsize[1, :], (0, 255, 0), 2)
            # cv2.putText(frame1, 'Detection Boundary', (410, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_4)
            # frame1 = frame1[self.top_down_fabric_cropsize[0, 1]:self.top_down_fabric_cropsize[1, 1], self.top_down_fabric_cropsize[0, 0]:self.top_down_fabric_cropsize[1, 0]]
            cv2.imshow('frame1', frame1)

            # 2D Fabric feature detection
            if self.D405_use_flag:
                left_color_image = self.D405.get_aligned_images(pipeline=self.D405.pipelines["left_D405"])[0]
                right_color_image = self.D405.get_aligned_images(pipeline=self.D405.pipelines["right_D405"])[0]

                try:
                    if self.display_state == 0:
                        pass

                    elif self.display_state == 1:
                        # cv2.rectangle(left_color_image, self.left_top_fabric_cropsize[0, :], self.left_top_fabric_cropsize[1, :], (0, 0, 255), 2)
                        # cv2.rectangle(right_color_image, self.right_top_fabric_cropsize[0, :], self.right_top_fabric_cropsize[1, :], (0, 0, 255), 2)

                        self.left_top_fabric_corner = find_FabricFeature(left_color_image, cropsize=self.left_top_fabric_cropsize, thrValue=160, findTop=True, side='left')[0]
                        self.right_top_fabric_corner = find_FabricFeature(right_color_image, cropsize=self.right_top_fabric_cropsize, thrValue=130, findTop=True, side='right')[0]

                    elif self.display_state == 2:
                        self.left_top_fabric_contour = find_FabricFeature(left_color_image, cropsize=self.left_top_fabric_cropsize, thrValue=160, findTop=True)[1]
                        self.right_top_fabric_contour = find_FabricFeature(right_color_image, cropsize=self.right_top_fabric_cropsize, thrValue=130, findTop=True)[1]

                        # for i in range(np.size(self.left_top_fabric_contour, axis=0)):
                        #     cv2.circle(left_color_image, (self.left_top_fabric_contour[i, 0], self.left_top_fabric_contour[i, 1]), 2, (0, 200, 10), -1)
                        #
                        # for i in range(np.size(self.right_top_fabric_contour, axis=0)):
                        #     cv2.circle(right_color_image, (self.right_top_fabric_contour[i, 0], self.right_top_fabric_contour[i, 1]), 2, (0, 200, 10), -1)

                    elif self.display_state == 5:
                        self.left_top_fabric_corner = np.array([0, 0], dtype=np.int32)
                        self.left_top_fabric_contour = np.array([0, 0], dtype=np.int32).reshape(-1, 2)

                        self.right_top_fabric_corner = np.array([0, 0], dtype=np.int32)
                        self.right_top_fabric_contour = np.array([0, 0], dtype=np.int32).reshape(-1, 2)

                    else:
                        pass

                    dataTrans = self.left_top_fabric_corner.reshape(1, -1).squeeze(axis=0)
                    self.pub10.publish(dataTrans)

                    dataTrans = self.right_top_fabric_corner.reshape(1, -1).squeeze(axis=0)
                    self.pub12.publish(dataTrans)

                    self.left_down_fabric_corner = find_FabricFeature(left_color_image, cropsize=self.left_down_fabric_cropsize,
                                                                      thrValue=120, findTop=False, top_contour=self.left_top_fabric_contour, side='left')

                    self.right_down_fabric_corner = find_FabricFeature(right_color_image, cropsize=self.right_down_fabric_cropsize,
                                                                       thrValue=110, findTop=False, top_contour=self.right_top_fabric_contour, side='right')

                    dataTrans = self.left_down_fabric_corner.reshape(1, -1).squeeze(axis=0)
                    self.pub11.publish(dataTrans)

                    dataTrans = self.right_down_fabric_corner.reshape(1, -1).squeeze(axis=0)
                    self.pub13.publish(dataTrans)

                    # cv2.circle(left_color_image, (self.left_top_fabric_corner[0], self.left_top_fabric_corner[1]), 6, (0, 0, 255), -1)
                    # cv2.circle(right_color_image, (self.right_top_fabric_corner[0], self.right_top_fabric_corner[1]), 6, (0, 0, 255), -1)
                    #
                    # if self.display_state == 3:
                    #     cv2.rectangle(left_color_image, self.left_down_fabric_cropsize[0, :], self.left_down_fabric_cropsize[1, :], (0, 255, 0), 2)
                    #     cv2.rectangle(right_color_image, self.right_down_fabric_cropsize[0, :], self.right_down_fabric_cropsize[1, :], (0, 255, 0), 2)
                    #
                    #     cv2.circle(left_color_image, (self.left_down_fabric_corner[0], self.left_down_fabric_corner[1]), 6, (0, 255, 0), -1)
                    #     cv2.circle(right_color_image, (self.right_down_fabric_corner[0], self.right_down_fabric_corner[1]), 6, (0, 255, 0), -1)

                except BaseException:
                    pass
                    # dataTrans = np.array([0, 0]).reshape(1, -1).squeeze(axis=0)
                    # self.pub10.publish(dataTrans)
                    # self.pub11.publish(dataTrans)
                    # self.pub12.publish(dataTrans)
                    # self.pub13.publish(dataTrans)

                cv2.imshow('D405_left_color_image', left_color_image)
                cv2.imshow('D405_right_color_image', right_color_image)
                # left_right_frame = np.hstack((left_color_image, right_color_image))
                # cv2.imshow('left_right_frame', left_right_frame)

            keyboardVal = cv2.waitKey(1)
            if keyboardVal * 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
            elif keyboardVal == ord('s'):
                np.savetxt('top_fabric_shape_2D.txt', top_fabric_shape_2D, delimiter=',')
                np.savetxt('down_fabric_shape_2D.txt', down_fabric_shape_2D, delimiter=',')

                np.savetxt('top_fabric_shape_3D.txt', top_fabric_shape_3D, delimiter=',')
                np.savetxt('down_fabric_shape_3D.txt', down_fabric_shape_3D, delimiter=',')

                np.savetxt('top_fabric_sidepoints_2D.txt', top_fabric_sidepoints_2D, delimiter=',')
                np.savetxt('down_fabric_sidepoints_2D.txt', down_fabric_sidepoints_2D, delimiter=',')

                np.savetxt('top_fabric_sidepoints_3D.txt', top_fabric_sidepoints_3D, delimiter=',')
                np.savetxt('down_fabric_sidepoints_3D.txt', down_fabric_sidepoints_3D, delimiter=',')

            self.rate.sleep()

    def callback(self, sub1_message):
        assert isinstance(sub1_message, Int32)
        self.display_state = int(sub1_message.data)


if __name__ == '__main__':
    rospy.init_node('imgpro_fabric', anonymous=True)
    ip = ImageProcessing()
    ip.run()




# def get_corners(box):  # 这里本人项目yaw [-pi/4, 3*pi/4)，需要映射到[0, pi)
#     x = box[0]
#     y = box[1]
#     w = box[2]
#     l = box[3]
#     yaw = box[4]
#     if yaw < 0:  # 用来映射
#         yaw = yaw + np.pi
#
#     bev_corners = np.zeros((4, 2), dtype=np.float32)
#     cos_yaw = np.cos(yaw)
#     sin_yaw = np.sin(yaw)
#
#     bev_corners[0, 0] = (w / 2) * cos_yaw - (l / 2) * sin_yaw + x
#     bev_corners[0, 1] = (w / 2) * sin_yaw + (l / 2) * cos_yaw + y
#
#     bev_corners[1, 0] = (l / 2) * sin_yaw + (w / 2) * cos_yaw + x
#     bev_corners[1, 1] = (w / 2) * sin_yaw - (l / 2) * cos_yaw + y
#
#     bev_corners[2, 0] = (-w / 2) * cos_yaw - (-l / 2) * sin_yaw + x
#     bev_corners[2, 1] = (-w / 2) * sin_yaw + (-l / 2) * cos_yaw + y
#
#     bev_corners[3, 0] = (-l / 2) * sin_yaw + (-w / 2) * cos_yaw + x
#     bev_corners[3, 1] = (-w / 2) * sin_yaw - (-l / 2) * cos_yaw + y
#
#     return bev_corners


# def Obtain3DShape(shape_2D, aligned_depth_frame, depth_intrin):
#     if np.size(shape_2D) == 2:
#         shape_2D = np.reshape(shape_2D, [1, 2])
#
#     depth = np.zeros(np.size(shape_2D, axis=0), dtype=np.float)
#     shape_3D = np.zeros((np.size(shape_2D, axis=0), 3), dtype=np.float)
#
#     for i in range(np.size(shape_2D, axis=0)):
#         depth[i] = aligned_depth_frame.get_distance(shape_2D[i, 0], shape_2D[i, 1])
#         shape_3D[i, :] = rs.rs2_deproject_pixel_to_point(depth_intrin, [shape_2D[i, 0], shape_2D[i, 1]], depth[i])
#
#     zero_idx = np.where(shape_3D[:, 2] == 0)[0]
#     nonzero_idx = np.where(shape_3D[:, 2] != 0)[0]
#     shape_3D[zero_idx, :] = shape_3D[nonzero_idx[0], :]
#
#     return shape_3D, depth