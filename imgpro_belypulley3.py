#! /home/tom/miniconda3/envs/lassoing_env/bin/python
import sys
import message_filters
# from imgpro_general import points_sorting
from std_msgs.msg import Int32
from cameraClass_ROS import *
from cameraClass_API import *



class ImageProcessing():
    def __init__(self):        
        self.D405 = RealSenseD405Set(displaysize='small')

    def run(self):
        while not rospy.is_shutdown():           
            D405_1 = self.D405.get_aligned_images(pipeline=self.D405.pipelines["left_D405"])[0]
            D405_2 = self.D405.get_aligned_images(pipeline=self.D405.pipelines["right_D405"])[0]

            cv2.imshow('D405_1', D405_1)
            cv2.imshow('D405_2', D405_2)

            keyboardVal = cv2.waitKey(1)
            if keyboardVal & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    rospy.init_node('D405_NODE', anonymous=True)
    ip = ImageProcessing()
    ip.run()