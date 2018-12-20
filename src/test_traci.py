#!/usr/bin/env python
import sumoenv as env
import time
import random
import numpy as np
import logging
import actor_sumo as acs

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

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
logger.addHandler(handler)

global line
line = Twist()

my_env = env.TrafficEnv()
f = open("./logger.txt",'w')

rospy.init_node('listener',anonymous=True)
rate = rospy.Rate(200)
test = acs.Test()

for i_episode in range(1000000):
    # listener()
    s = my_env.reset()
    N_others = 12 * 10
    s_pre_others = np.zeros((N_others))
    s_pre_others2 = np.array(s[1] + s[2])
    for i in range(N_others):
        s_pre_others[i] = s_pre_others2[i % 12]

    s_sliding, s_others = s[0], s_pre_others
    ep_r = 0
    # fsm.Tick(command)
    while True:
        listener()
        if random.random()<0.999:
            action = test.choose_action(s_sliding,s_others)
        else:
            action = get_command()    
        #print("now_action",int(action))
        s, r, is_done, dist = my_env.step(action)

        #print("reward=",r)
        s_pre_others2 = np.array(s[1] + s[2])
        for i in range(N_others):
            s_pre_others[i] = s_pre_others2[i % 12]

        s_sliding_, s_others_ = s[0], s_pre_others

        """
        f.write(str(s_sliding))
        f.write("!")
        f.write(str(s_others))
        f.write("@")
        f.write(str(action))
        f.write("#")
        f.write(str(r))
        f.write("$")
        f.write(str(s_sliding_))
        f.write("%")
        f.write(str(s_others_))
        f.write("^")
        f.write(str(is_done))
        f.write("&")
        f.write("\n")
        """

        s_sliding = s_sliding_
        s_others = s_others_

        log = str(s_others)
        #print(log)

        ep_r += r

            #bt.learn()
        if is_done:
            print("Ep:", i_episode, "|Ep_r:", round(ep_r, 2))
            break

f.close()
