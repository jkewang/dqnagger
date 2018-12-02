#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from env import Env

import numpy as np
import random
import math

def callback(data):
    global line
    line = data

def listener():
    global line
    rospy.Subscriber("/turtle1/cmd_vel",Twist,callback)

def talker(netout):
    pub.publish(netout)

def get_command(): 
    if line.linear.x == 2:	
        return 0
    elif line.linear.x == -2:
        return 1
    elif line.angular.z == 2:
        return 2
    else:
        return 3

global line
line = Twist()

if __name__ == '__main__':
    rospy.init_node('listener',anonymous=True)
    rate = rospy.Rate(200)
    pub = rospy.Publisher('/gazebo/set_model_state',ModelState, queue_size=1)
    multi_env = Env()
    while(True):
        listener()
        tele_action = get_command()
        multi_env.step(tele_action)
        for car in multi_env.Cars:
            talker(car.modelstate)
        for car in multi_env.Waiting_cars:
            talker(car.modelstate)
        
        rate.sleep()

















