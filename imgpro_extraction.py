import pyrealsense2 as rs
import numpy as np
import cv2
from copy import deepcopy
from time import time


def nothing(x):
    pass


def extraction_hsv(img, hsv_value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_max = hsv_value[0]
    s_max = hsv_value[1]
    v_max = hsv_value[2]

    h_min = hsv_value[3]
    s_min = hsv_value[4]
    v_min = hsv_value[5]

    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    points = np.hstack((np.where(closing == 255)[1].reshape((-1, 1)), np.where(closing == 255)[0].reshape((-1, 1))))
    for i in range(np.size(points, 0)):
        cv2.circle(img, (points[i, 0], points[i, 1]), 1, (0, 255, 0), -1)

    return img, closing


def extraction_thr(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # kernel = np.ones((5,5), np.uint8)  
    # dilated = cv2.dilate(closing, kernel, iterations=5)


    return img, closing


if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()

    # config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    hsvSel = False

    if hsvSel:
        cv2.namedWindow('closing')
<<<<<<< HEAD
        cv2.createTrackbar('h', 'closing', 54, 180, nothing)
        cv2.createTrackbar('s', 'closing', 63, 255, nothing)
        cv2.createTrackbar('v', 'closing', 84, 255, nothing)
=======
        cv2.createTrackbar('h_max', 'closing', 24, 180, nothing)
        cv2.createTrackbar('s_max', 'closing', 129, 255, nothing)
        cv2.createTrackbar('v_max', 'closing', 57, 255, nothing)

        cv2.createTrackbar('h_min', 'closing', 24, 180, nothing)
        cv2.createTrackbar('s_min', 'closing', 129, 255, nothing)
        cv2.createTrackbar('v_min', 'closing', 57, 255, nothing)
>>>>>>> a61417749b64e8ac1a7d5e0af6b0aa3419422410
    else:
        cv2.namedWindow('closing')
        cv2.createTrackbar('thresh', 'closing', 125, 255, nothing)

    while True:
        tic = time()
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if hsvSel:
            h_max = cv2.getTrackbarPos('h_max', 'closing')
            s_max = cv2.getTrackbarPos('s_max', 'closing')
            v_max = cv2.getTrackbarPos('v_max', 'closing')

            h_min = cv2.getTrackbarPos('h_min', 'closing')
            s_min = cv2.getTrackbarPos('s_min', 'closing')
            v_min = cv2.getTrackbarPos('v_min', 'closing')

            img, closing = extraction_hsv(color_image, [h_max, s_max, v_max, h_min, s_min, v_min])
        else:
            thr = cv2.getTrackbarPos('thresh', 'closing')
            img, closing = extraction_thr(color_image, threshold=thr)

        cv2.putText(img, '%d Hz' % (1.0 / (time() - tic)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_4)
        cv2.imshow('img', img)
        cv2.imshow('closing', closing)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break


