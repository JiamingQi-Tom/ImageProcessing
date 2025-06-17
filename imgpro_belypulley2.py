#! /home/tom/miniconda3/envs/lassoing_env/bin/python
import sys
import message_filters
# from imgpro_general import points_sorting
from std_msgs.msg import Int32
from cameraClass_ROS import *
from cameraClass_API import *



class ImageProcessing(RealSenseRosSet):
    def __init__(self):
        super(ImageProcessing, self).__init__()        

    def run(self):
        while not rospy.is_shutdown():
            frame1 = deepcopy(self.color_image)
            cv2.imshow('frame1', frame1)

            keyboardVal = cv2.waitKey(1)
            if keyboardVal & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    rospy.init_node('D455_NODE', anonymous=True)
    ip = ImageProcessing()
    ip.run()