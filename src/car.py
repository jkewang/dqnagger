#!/usr/bin/env python
from gazebo_msgs.msg import ModelState
import random

class Car(object):
    def __init__(self,name,x,y,vx):
        self.name = name
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = 0
        self.del_t = 0.01
        self.lastx = x
        self.lasty = y
        self.lastaction = 0
        self.state = []
        self.state_ = []
        self.ep_r = 0
        self.done = 0
        self.waiting = 0
        self.MAXSPEED = (float(random.randint(0,400))/100)+4
        self.modelstate = ModelState()
        self.modelstate.model_name = self.name
        self.modelstate.pose.position.x = self.x
        self.modelstate.pose.position.y = self.y
        self.modelstate.pose.position.z = 0.55

    def reset(self,x,y,vx):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = 0
        self.lastx = x
        self.lasty = y
        self.lastaction = 0
        self.state = []
        self.state_ = []
        self.ep_r = 0
        self.done = 0
        self.waiting = 0
        self.MAXSPEED = (float(random.randint(0,400))/100)+4
        self.modelstate = ModelState()
        self.modelstate.model_name = self.name
        self.modelstate.pose.position.x = self.x
        self.modelstate.pose.position.y = self.y
        self.modelstate.pose.position.z = 0.55

        return x,y,vx,self.vy

    def update_v(self,a):
        if(a==0 and self.vx > -self.MAXSPEED):
            self.vx -= 0.03
            self.vy = 0
        if(a==0 and self.vy !=0):
            self.vy = 0
        if(a==1 and self.vx < -2):
            self.vx += 0.03
            self.vy = 0
        if(a==2 and self.vy>-2):
            self.vx = self.vx
            self.vy -= 0.02
        if(a==3 and self.vy<2):
            self.vx =self.vx
            self.vy += 0.02

        self.lastaction = a

    def play(self):
        self.lastx = self.x
        self.lasty = self.y
        self.x = self.x + self.vx * self.del_t
        self.y = self.y + self.vy * self.del_t
        self.modelstate.pose.position.x = self.x
        self.modelstate.pose.position.y = self.y

        return self.x,self.y,self.vx,self.vy
