#!/usr/bin/env python3
import rospy
import numpy as np
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R


# def xyzrgb2pointcloud2(points, colors, frame_id='world'):
#     points = np.asarray(points)
#     colors = np.asarray(colors)
#     rowNUm = np.size(points, axis=0)
#
#     rgb = np.zeros((rowNUm, 1))
#     for i in range(rowNUm):
#         r = int(colors[i, 0])
#         g = int(colors[i, 1])
#         b = int(colors[i, 2])
#         a = 255
#         temp = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
#         rgb[i] = temp
#     points = np.hstack((points, rgb))
#
#     fields = [PointField('x', 0, PointField.FLOAT32, 1),
#               PointField('y', 4, PointField.FLOAT32, 1),
#               PointField('z', 8, PointField.FLOAT32, 1),
#               PointField('rgba', 12, PointField.UINT32, 1)]
#
#     header = Header()
#     header.stamp = rospy.Time.now()
#     header.frame_id = frame_id
#     pointcloud = point_cloud2.create_cloud(header, fields=fields, points=points)
#
#     return pointcloud


def xyzrgb2pointcloud2(points,
                       idx=None,
                       color_image=None,
                       color_fixed=False,
                       colors=np.array([[100, 100, 100]]),
                       frame_id='world'):

    points = np.asarray(points)
    if np.size(points) == 3:
        points = points.reshape(1, -1)

    C = np.zeros((np.size(points, 0), 4), dtype=np.uint8)

    if not color_fixed:
        C[:, 0:3] = color_image[idx[:, 1], idx[:, 0]]
        C[:, 3] = 255 * np.ones(np.size(points, axis=0))

    else:
        if len(colors) == 3:
            for i in range(np.size(points, axis=0)):
                C[i, 0] = np.array(colors[2]).astype(np.uint8)
                C[i, 1] = np.array(colors[1]).astype(np.uint8)
                C[i, 2] = np.array(colors[0]).astype(np.uint8)
                C[i, 3] = 255
        else:
            for i in range(np.size(points, axis=0)):
                C[i, 0] = np.array(colors[i, 2]).astype(np.uint8)
                C[i, 1] = np.array(colors[i, 1]).astype(np.uint8)
                C[i, 2] = np.array(colors[i, 0]).astype(np.uint8)
                C[i, 3] = 255

    # ----------------------------------------------------------------------------------------- #
    C = C.view("uint32")

    pointsColor = np.zeros((points.shape[0], 1), dtype={"names": ("x", "y", "z", "rgba"), "formats": ("f4", "f4", "f4", "u4")})
    points = points.astype(np.float32)

    pointsColor["x"] = points[:, 0].reshape((-1, 1))
    pointsColor["y"] = points[:, 1].reshape((-1, 1))
    pointsColor["z"] = points[:, 2].reshape((-1, 1))
    pointsColor["rgba"] = C

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    msg = PointCloud2()
    msg.header = header

    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = points.shape[0]

    msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1)]

    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = pointsColor.tostring()

    return msg


def transform_shape(shape, Ts):
    N = np.size(shape, axis=0)
    if N == 1:
        shape = shape.reshape(-1, 1)
        shape = np.vstack((shape, 1))

        shape = Ts.dot(shape)
        shape = shape[0:3, 0]
    else:
        shape = np.hstack((shape, np.ones((N, 1))))
        shape = np.transpose(Ts.dot(np.transpose(shape)))
        shape = shape[:, 0:3]

    return shape


def display_tf(transformationMatrix, frame_id="frame1"):
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()

    quatern = R.from_matrix(transformationMatrix[0:3, 0:3]).as_quat()

    static_transformStamped.header.frame_id = "world"
    static_transformStamped.child_frame_id = frame_id
    static_transformStamped.transform.translation.x = transformationMatrix[0, 3]
    static_transformStamped.transform.translation.y = transformationMatrix[1, 3]
    static_transformStamped.transform.translation.z = transformationMatrix[2, 3]
    static_transformStamped.transform.rotation.x = quatern[0]
    static_transformStamped.transform.rotation.y = quatern[1]
    static_transformStamped.transform.rotation.z = quatern[2]
    static_transformStamped.transform.rotation.w = quatern[3]
    static_transformStamped.header.stamp = rospy.Time.now()
    broadcaster.sendTransform(static_transformStamped)


if __name__ == '__main__':
    rospy.init_node('pcl2_pub_example')
    pcl_pub = rospy.Publisher("/point_cloud2", PointCloud2, queue_size=2)
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        points = (np.random.rand(100, 3) - 0.5) * 2 * 0.2 + 1
        pc2 = xyzrgb2pointcloud2(points, color_fixed=True, colors=np.array([255, 0, 0]), frame_id='world')

        display_tf(np.eye(4, dtype=np.float))

        rospy.loginfo('I am sending a message')
        pcl_pub.publish(pc2)
        rate.sleep()


# sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
# sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u
# sudo apt-get install librealsense2-dkms
# sudo apt-get install librealsense2-utils
# sudo apt-get install librealsense2-dev
# sudo apt-get install librealsense2-dbg
