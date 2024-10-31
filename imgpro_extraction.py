import pyrealsense2 as rs
import numpy as np
import cv2
from copy import deepcopy
from time import time


def nothing(x):
    pass


def extraction_hsv(img, hsv_value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv_value[0]
    s = hsv_value[1]
    v = hsv_value[2]

    lower_hsv = np.array([h, s, v])
    upper_hsv = np.array([h + 30, 255, 255])
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

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return img, closing


if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()

    # config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)

    # profile = pipeline.start(config)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    hsvSel = True

    if hsvSel:
        cv2.namedWindow('closing')
        cv2.createTrackbar('h', 'closing', 24, 180, nothing)
        cv2.createTrackbar('s', 'closing', 129, 255, nothing)
        cv2.createTrackbar('v', 'closing', 57, 255, nothing)
    else:
        cv2.namedWindow('closing')
        cv2.createTrackbar('thresh', 'closing', 0, 255, nothing)

    while True:
        tic = time()
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if hsvSel:
            h = cv2.getTrackbarPos('h', 'closing')
            s = cv2.getTrackbarPos('s', 'closing')
            v = cv2.getTrackbarPos('v', 'closing')

            img, closing = extraction_hsv(color_image, [h, s, v])
        else:
            thr = cv2.getTrackbarPos('thresh', 'closing')
            img, closing = extraction_thr(color_image, threshold=thr)

        cv2.putText(img, '%d Hz' % (1.0 / (time() - tic)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_4)
        cv2.imshow('img', img)
        cv2.imshow('closing', closing)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break




















