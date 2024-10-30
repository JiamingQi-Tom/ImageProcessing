# coding=utf-8
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from copy import deepcopy
from ClusterAlgorithms import *
from copy import *


def find_area(origin, threshold=np.array([106, 68, 45]), hsv_flag=True):
    if hsv_flag:
        h, s, v = threshold[0], threshold[1], threshold[2]
        lower_hsv = np.array([h, s, v])
        upper_hsv = np.array([h + 30, 255, 255])

        hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    else:
        gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, np.uint64(threshold), 255, cv2.THRESH_BINARY)

        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((1, 1), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    [x, y] = np.where(closing == 255)
    points = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    return points, closing


def points_sorting(points, base):
    N = np.size(points, axis=0)
    ordered = np.zeros_like(points)
    distance = []
    for i in range(N):
        for j in range(np.size(points, axis=0)):
            distance.append(np.linalg.norm(base - points[j, :], ord=2))
        index = np.argmin(np.array(distance))
        ordered[i, :] = points[index, :]
        points = np.delete(points, index, axis=0)
        distance = []
        base = deepcopy(ordered[i, :])

    return ordered


def find_single_red(origin):
    hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)

    h, s, v = 151, 126, 65
    lower_hsv = np.array([h, s, v])
    upper_hsv = np.array([h + 30, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy, = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    contour = contours[np.argmax(np.array(area))]
    contour = np.squeeze(contour, axis=1)

    x, y, w, h = cv2.boundingRect(contour)
    red = np.array([x + w / 2, y + h / 2]).astype(int)
    return red


def find_double_purple(origin, color):
    closing = find_area(origin, color)[1]

    # method 1
    # img, contours, hierarchy, = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # area = []
    # for i in range(len(contours)):
    #     area.append(cv2.contourArea(contours[i]))
    #
    # area = array(area)
    # top_k_idx = area.argsort()[::-1][0:2]
    #
    # contour1 = squeeze(contours[top_k_idx[0]], axis=1)
    # contour2 = squeeze(contours[top_k_idx[1]], axis=1)
    #
    # center1 = mean(contour1, axis=0).astype(int)
    # center2 = mean(contour2, axis=0).astype(int)
    #
    # if center1[1] > center2[1]:
    #     left_red = center1
    #     right_red = center2
    # else:
    #     left_red = center2
    #     right_red = center1
    #
    # newbinary = zeros((origin.shape[1], origin.shape[0]), dtype=uint8)
    # newbinary = cv2.fillPoly(newbinary, [contour1], (255))
    # newbinary = cv2.fillPoly(newbinary, [contour2], (255))
    #
    # [areax, areay] = where(newbinary == 255)
    # area = hstack((areax.reshape(-1, 1), areay.reshape(-1, 1)))

    # method 2
    closing1 = deepcopy(closing[100:400, 110:270])
    contours, hierarchy, = cv2.findContours(closing1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(np.array(area))
    contour1 = contours[max_idx]
    contour1 = np.squeeze(contour1, axis=1)
    contour1 += np.array([110, 100])
    center1 = np.mean(contour1, axis=0).astype(int)

    closing2 = deepcopy(closing[100:400, 270:500])
    contours, hierarchy, = cv2.findContours(closing2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(np.array(area))
    contour2 = contours[max_idx]
    contour2 = np.squeeze(contour2, axis=1)
    contour2 += np.array([270, 100])
    center2 = np.mean(contour2, axis=0).astype(int)

    newbinary = np.zeros((origin.shape[1], origin.shape[0]), dtype=np.uint8)
    newbinary = cv2.fillPoly(newbinary, [contour1], (255))
    newbinary = cv2.fillPoly(newbinary, [contour2], (255))
    [areax, areay] = np.where(newbinary == 255)
    area = np.hstack((areax.reshape(-1, 1), areay.reshape(-1, 1)))

    return np.asarray(center1), np.asarray(center2), area


def find_double_yellow(origin, left, right, color):
    closing = find_area(origin, color)[1]

    closing1 = deepcopy(closing[100:400, 110:left[0]])
    contours, hierarchy, = cv2.findContours(closing1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(np.array(area))
    contour1 = contours[max_idx]
    contour1 = np.squeeze(contour1, axis=1)
    contour1 += + np.array([110, 100])
    center1 = np.mean(contour1, axis=0).astype(int)

    closing2 = deepcopy(closing[100:400, right[0]:500])
    contours, hierarchy, = cv2.findContours(closing2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(np.array(area))
    contour2 = contours[max_idx]
    contour2 = np.squeeze(contour2, axis=1)
    contour2 += np.array([right[0], 100])
    center2 = np.mean(contour2, axis=0).astype(int)

    newbinary = np.zeros((origin.shape[1], origin.shape[0]), dtype=np.uint8)
    newbinary = cv2.fillPoly(newbinary, [contour1], (255))
    newbinary = cv2.fillPoly(newbinary, [contour2], (255))
    [areax, areay] = np.where(newbinary == 255)
    area = np.hstack((areax.reshape(-1, 1), areay.reshape(-1, 1)))

    center1 = np.asarray(center1)
    center2 = np.asarray(center2)

    return center1, center2, area


def calibration_point(origin, color):
    closing = find_area(origin, color)[1]

    closing = deepcopy(closing[100:400, 110:540])
    img, contours, hierarchy, = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_idx = np.argmax(np.array(area))
    contour = contours[max_idx]
    contour = np.squeeze(contour, axis=1)
    contour += + np.array([110, 100])
    center = np.mean(contour, axis=0).astype(int)

    return center


# def find_yellow_surface(origin, color, fixedNum=10):
#     # yellow sponge
#     closing = find_area(origin, color)[1]
#     contours, hierarchy, = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#     area = []
#     for i in range(len(contours)):
#         area.append(cv2.contourArea(contours[i]))
#
#     max_idx = np.argmax(np.array(area))
#     contour = contours[max_idx]
#     contour = np.squeeze(contour, axis=1)
#
#     newbinary = np.zeros((origin.shape[1], origin.shape[0]), dtype=np.uint8)
#     newbinary = cv2.fillPoly(newbinary, [contour], (255))
#
#     surface = np.hstack((np.where(newbinary == 255)[0].reshape((-1, 1)), np.where(newbinary == 255)[1].reshape((-1, 1))))
#
#     f = FPS(surface)
#     surface = f.comput_fps(fixedNum).astype(int)
#     surface = surface_sorting(surface)
#
#     center = np.mean(surface, axis=0).astype(int)
#     surface -= np.array([center[0], center[1]])
#     bias = 5
#     for i in range(np.size(surface, axis=0)):
#         if surface[i, 0] >= 0 and surface[i, 1] >= 0:
#             surface[i, 0] -= bias
#             surface[i, 1] -= bias
#         elif surface[i, 0] <= 0 and surface[i, 1] >= 0:
#             surface[i, 0] += bias
#             surface[i, 1] -= bias
#         elif surface[i, 0] <= 0 and surface[i, 1] <= 0:
#             surface[i, 0] += bias
#             surface[i, 1] += bias
#         elif surface[i, 0] >= 0 and surface[i, 1] <= 0:
#             surface[i, 0] -= bias
#             surface[i, 1] += bias
#     surface += np.array([center[0], center[1]])
#
#     # yellow plane
#     # yellow, closing = find_area(origin, color)
#
#     # closing = closing[77:480, 64:640]
#     # [x, y] = where(closing == 255)
#     # yellow = hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
#
#     # f = FPS(yellow)
#     # surface = f.comput_fps(fixNum)
#     # surface = surface + [77, 64]
#
#     # surface = centerline_ordered_fixed_GMM(yellow, fixedNum=10)
#     # surface = surface + [77, 64]
#
#     # newimg = zeros((origin.shape[0], origin.shape[1], 3), dtype=uint8)
#     # newimg = cv2.fillPoly(newimg, [contour], (255, 255, 255))
#     # gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
#     # ret, thresh = cv2.threshold(gray, 56, 255, cv2.THRESH_BINARY)
#
#     return surface


# def surface_sorting(points):
#     N = np.size(points, 0)
#     ordered = np.zeros((N, 2), dtype=int)
#     k1 = 0
#     minu = min(points[:, 0])
#     maxu = max(points[:, 0])
#     for i in range(minu, maxu + 1):
#         index = np.where(points[:, 0] == i)
#
#         if np.size(index[0]) == 1:
#             ordered[k1, :] = points[index[0][0], :]
#             k1 += 1
#             points = np.delete(points, index[0][0], axis=0)
#
#         if np.size(index[0]) > 1:
#             index = index[0][np.argsort(points[index[0], 1])]
#             for j in range(np.size(index)):
#                 ordered[k1 + j, :] = points[index[j], :]
#
#             k1 += np.size(index)
#             points = np.delete(points, index, axis=0)
#
#         if k1 == N:
#             ordered = ordered.astype(int)
#             break
#
#     ordered = ordered.astype(int)
#     return ordered








# def find_contour_middle1(origin, color, fixedNum=20, start=[245, 61]):
#     closing = find_area(origin, color)[1]
#     closing = deepcopy(closing[100:400, 180:500])
#
#     # Extract all contours
#     contours, hierarchy, = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     area = []
#     for i in range(len(contours)):
#         area.append(cv2.contourArea(contours[i]))
#
#     area = np.asarray(area)
#     area_ = deepcopy(area)
#     area_[np.argmax(area_)] = np.min(area)
#     idx_1st = np.argmax(area)
#     idx_2rd = np.argmax(area_)
#
#     contour_1st = contours[idx_1st]
#     contour_1st = np.squeeze(contour_1st, axis=1)
#
#     contour_2rd = contours[idx_2rd]
#     contour_2rd = np.squeeze(contour_2rd, axis=1)
#
#     newbinary = np.zeros((closing.shape[1], closing.shape[0]), dtype=np.uint8)
#     newbinary = cv2.fillPoly(newbinary, [contour_1st], (255))
#     newbinary = cv2.fillPoly(newbinary, [contour_2rd], (0))
#
#     # Extract centerline
#     thinned = cv2.ximgproc.thinning(newbinary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
#     centerline = np.hstack((np.where(thinned == 255)[1].reshape((-1, 1)), np.where(thinned == 255)[0].reshape((-1, 1))))
#     centerline += np.array([180, 100])
#     centerline[:, [0, 1]] = centerline[:, [1, 0]]
#
#     # Down-sampling
#     f = FPS(centerline)
#     centerline = f.comput_fps(fixedNum)
#
#     # Sorting
#     base = start
#     distance = []
#     N = np.size(centerline, 0)
#     ordered = np.zeros((N, 2), dtype=int)
#     for i in range(N):
#         for j in range(np.size(centerline, 0)):
#             distance.append(np.linalg.norm(base - centerline[j, :], ord=2))
#         index = np.argmin(np.array(distance))
#         ordered[i, :] = centerline[index, :]
#         centerline = np.delete(centerline, index, axis=0)
#         distance = []
#         base = ordered[i, :]
#
#     return ordered





# def find_pillow(origin, color, fixedNum=20, start=[245, 61]):
#     green, closing = find_area(origin, color)
#
#     img, contours, hierarchy, = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     area = []
#     for i in range(len(contours)):
#         area.append(cv2.contourArea(contours[i]))
#
#     max_idx = np.argmax(np.array(area))
#     contour = contours[max_idx]
#     contour = np.squeeze(contour, axis=1)
#     contour[:, [0, 1]] = contour[:, [1, 0]]
#
#     # Determine perimeter
#     perimeter = cv2.arcLength(contour, True)
#
#     # Down-sampling
#     newContour = np.zeros((fixedNum, 2))
#     interval = np.floor(perimeter / float(fixedNum - 1)) - 0
#     try:
#         k1 = 1
#         k2 = 1
#         distance = 0
#         newContour[0][:] = contour[0][:]
#
#         while k1 <= fixedNum + 1:
#             if distance < interval:
#                 distance += np.linalg.norm(contour[k2, :] - contour[k2 - 1, :])
#                 k2 += 1
#             else:
#                 newContour[k1][:] = contour[k2][:]
#                 k1 += 1
#                 distance = 0
#     except BaseException:
#         pass
#
#     newContour = newContour.astype(int)
#
#     try:
#         zeroindex = np.where(newContour[:, 0] == 0)[0][0]
#         newContour[zeroindex:fixedNum, :] = newContour[zeroindex - 1, :]
#     except BaseException:
#         pass
#
#     newContour = np.flipud(newContour)
#
#     # Reorder newContour
#     distance = np.zeros(fixedNum, dtype=float)
#     for i in range(np.size(newContour, 0)):
#         distance[i] = np.linalg.norm(newContour[i, :] - start)
#
#     index = np.argmin(distance)
#     newContour = np.vstack((newContour[index:, :], newContour[0:index, :]))
#
#     return newContour


# def find_contour_middle(origin, color, fixedNum=20, start=[245, 61]):
#     # closing = find_area(origin, color)[1]
#     # closing = deepcopy(closing[100:400, 110:500])
#
#     gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray, 56, 255, cv2.THRESH_BINARY_INV)
#     closing = deepcopy(thresh[100:400, 110:500])
#
#     img, contours, hierarchy, = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     area = []
#     for i in range(len(contours)):
#         area.append(cv2.contourArea(contours[i]))
#
#     max_idx = np.argmax(np.array(area))
#     contour = contours[max_idx]
#     contour = np.squeeze(contour, axis=1)
#     contour[:, [0, 1]] = contour[:, [1, 0]]
#
#     contour += np.array([100, 110])
#
#     center = np.mean(contour, axis=0).astype(int)
#     contour -= np.array([center[0], center[1]])
#     bias = 1
#     for i in range(np.size(contour, axis=0)):
#         if contour[i, 0] >= 0 and contour[i, 1] >= 0:
#             contour[i, 0] -= bias
#             contour[i, 1] -= bias
#         elif contour[i, 0] <= 0 and contour[i, 1] >= 0:
#             contour[i, 0] += bias
#             contour[i, 1] -= bias
#         elif contour[i, 0] <= 0 and contour[i, 1] <= 0:
#             contour[i, 0] += bias
#             contour[i, 1] += bias
#         elif contour[i, 0] >= 0 and contour[i, 1] <= 0:
#             contour[i, 0] -= bias
#             contour[i, 1] += bias
#     contour += np.array([center[0], center[1]])
#
#     # Determine perimeter
#     perimeter = cv2.arcLength(contour, True)
#
#     # Down-sampling
#     newContour = np.zeros((fixedNum, 2))
#     interval = np.floor(perimeter / float(fixedNum - 1)) - 0
#     try:
#         k1 = 1
#         k2 = 1
#         distance = 0
#         newContour[0][:] = contour[0][:]
#
#         while k1 <= fixedNum + 1:
#             if distance < interval:
#                 distance += np.linalg.norm(contour[k2, :] - contour[k2 - 1, :])
#                 k2 += 1
#             else:
#                 newContour[k1][:] = contour[k2][:]
#                 k1 += 1
#                 distance = 0
#     except BaseException:
#         pass
#
#     newContour = newContour.astype(int)
#
#     try:
#         zeroindex = np.where(newContour[:, 0] == 0)[0][0]
#         newContour[zeroindex:fixedNum, :] = newContour[zeroindex - 1, :]
#     except BaseException:
#         pass
#
#     newContour = np.flipud(newContour)
#
#     # Reorder newContour
#     distance = np.zeros(fixedNum, dtype=float)
#     for i in range(np.size(newContour, 0)):
#         distance[i] = np.linalg.norm(newContour[i, :] - start)
#
#     index = np.argmin(distance)
#     newContour = np.vstack((newContour[index:, :], newContour[0:index, :]))
#
#     return newContour





# def find_yellow_ball(origin):
#     hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
#     h, s, v = 16, 70, 50
#     lower_marker_hsv = np.array([h, s, v])
#     upper_marker_hsv = np.array([h + 20, 255, 255])
#     mask = cv2.inRange(hsv, lower_marker_hsv, upper_marker_hsv)
#
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#
#     [markerx, markery] = np.where(closing == 255)
#     marker = (np.round(np.mean(markery)).astype(int), np.round(np.mean(markerx)).astype(int))
#
#     return np.asarray(marker)


# def find_contour(origin, frame):
#     hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
#     h = cv2.getTrackbarPos('h', 'res')
#     s = cv2.getTrackbarPos('s', 'res')
#     v = cv2.getTrackbarPos('v', 'res')
#     lower_blue = np.array([h - 10, s, v])
#     upper_blue = np.array([h + 10, 255, 255])
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     # res = cv2.bitwise_and(frame1, frame1, mask=mask)
#     kernel = np.ones((5, 5), np.uint8)
#     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#     # laplacian = cv2.Laplacian(closing, cv2.CV_64F)
#     blur = cv2.GaussianBlur(closing, (5, 5), 0)
#     canny = cv2.Canny(blur, 100, 200)
#
#     contoure_posi = np.hstack((np.where(canny == 255)[1].reshape((-1, 1)), np.where(canny == 255)[0].reshape((-1, 1))))
#     # np.savetxt('D:\\GitHub\\exercise\\opencv\\data\\contoure_posi.txt', contoure_posi, fmt='%f', delimiter=',')
#     for i in range(np.size(contoure_posi, 0)):
#         cv2.circle(frame, (contoure_posi[i][0], contoure_posi[i][1]), 2, (255, 0, 0), -1)
#
#     return contoure_posi


# def find_center_line(origin, marker):
#     # get ROI area
#     gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
#     ROI = thresh[0:-1, :marker[0] - 20]
#
#     # get centerline
#     kernel = np.ones((1, 1), np.uint8)
#     opening = cv2.morphologyEx(ROI, cv2.MORPH_OPEN, kernel)
#     thinned = cv2.ximgproc.thinning(opening, thinningType=cv2.ximgproc.THINNING_GUOHALL)
#     centerline = np.hstack((np.where(thinned == 255)[1].reshape((-1, 1)), np.where(thinned == 255)[0].reshape((-1, 1))))
#
#     return np.asarray(centerline)


# def find_black_rod_centerline(origin, red_area, yellow_area):
#     # Change into grayscale
#     gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
#
#     # Fill red/yellow area
#     for i in range(np.size(red_area, 0)):
#         gray[red_area[i, 0], red_area[i, 1]] = 255
#
#     for i in range(np.size(yellow_area, 0)):
#         gray[yellow_area[i, 0], yellow_area[i, 1]] = 255
#
#     # Threshold Binary
#     ret, thresh = cv2.threshold(gray, 56, 255, cv2.THRESH_BINARY_INV)
#
#     # Calculate contours
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#     img, contours, hierarchy, = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#     # Find the contour with the biggest area, namely, black rod
#     area = []
#     for i in range(len(contours)):
#         area.append(cv2.contourArea(contours[i]))
#
#     max_idx = np.argmax(np.array(area))
#     contour = contours[max_idx]
#     contour = np.squeeze(contour, axis=1)
#
#     newimg = np.zeros((480, 640, 3), dtype=np.uint8)
#     newimg = cv2.fillPoly(newimg, [contour], (255, 255, 255))
#     gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
#
#     thinned = cv2.ximgproc.thinning(gray, thinningType=cv2.ximgproc.THINNING_GUOHALL)
#     centerline = np.hstack((np.where(thinned == 255)[1].reshape((-1, 1)), np.where(thinned == 255)[0].reshape((-1, 1))))
#
#     return centerline
