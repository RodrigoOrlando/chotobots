# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:16:06 2016

@author: rod
"""

import cv2
import numpy as np
import traceback,sys
from sender import Sender

class ChotobotTracker(object):
    def __init__(self,h,s,v):
        '''
        Chotobot Tracker from a camera, estimating the pose (angle, point reference)
        mobile robot with 3 leds above the robot.
        '''
        self.kernel = np.ones((6,6),np.uint8)
        #self.h,self.s,self.v = 30,60,255
        self.h,self.s,self.v = h,s,v
        self.min_area = 5
        self.mask=None
        self.img=None
        self.frame=None
        self.continue_=False
        #
        self.angle=0
        self.pos=0
    def nothing(self,x):
        pass
    def start(self):
        '''Start capture'''
        self.cap = cv2.VideoCapture(1)
        self.continue_=True
        cv2.namedWindow('Camara')
        cv2.namedWindow('mask')
        cv2.createTrackbar('Hue','mask',0,180,self.nothing)
        cv2.createTrackbar('Sat','mask',0,255,self.nothing)
        cv2.createTrackbar('Val','mask',0,255,self.nothing)
        cv2.createTrackbar('mA','mask',1,500,self.nothing)
        cv2.setTrackbarPos('Hue','mask',self.h)
        cv2.setTrackbarPos('Sat','mask',self.s)
        cv2.setTrackbarPos('Val','mask',self.v)
        cv2.setTrackbarPos('mA','mask',self.min_area)
        self.sender=Sender()
        self.sender("/dev/ttyUSB1")
    def trackbarReading(self):
        self.h = cv2.getTrackbarPos('Hue','mask')
        self.s = cv2.getTrackbarPos('Sat','mask')
        self.v = cv2.getTrackbarPos('Val','mask')
        self.min_area = cv2.getTrackbarPos('mA','mask')
    def process(self):
        '''
        Start the tracking algorithm, iteratively capturing the frames from the 
        previous capture object initialized by start()
        '''
        while(self.continue_):
            _, self.frame = self.cap.read()
        
            #converting to HSV
            hsv = cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)
        
            self.trackbarReading()
            # Normal masking algorithm
            lower_blue = np.array([self.h,self.s,self.v-50])
            upper_blue = np.array([self.s+25,255,255])
        
            self.mask = cv2.inRange(hsv,lower_blue, upper_blue)
        
            result = cv2.bitwise_and(self.frame,self.frame,mask = self.mask)
            self.mask = cv2.bitwise_and(self.frame,self.frame,mask = self.mask)
            self.img=result
            
            filtered = cv2.dilate(self.img, self.kernel)
            filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, self.kernel,40)
            cv2.GaussianBlur(filtered, (5, 5), 10)
            _, filtered = cv2.threshold(filtered,155,255,cv2.THRESH_BINARY) 
            
            filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            contours,hierarchy = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGRA)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
            areas = [cv2.contourArea(c) for c in contours]
            points=list()
            i = 0
            for extension in areas:
                if extension > self.min_area:
                    actual = contours[i]
                    M = cv2.moments(actual)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    points.append([cx,cy])
                    #cv2.putText(img, str(cx)+','+str(cy),(cx-10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255))
                    cv2.circle(self.img,(cx,cy),1,(60,255,255),1,8,0)
                    cv2.drawContours(self.img,[actual],0,(255,0,255),1)
                i = i+1
            if len(points)==3:
                try:
                    self.getTrianglePose(points)
                    if self.angle<=255:
                        self.sender.threadedSend(round(90-self.angle))
                except:
                    traceback.print_exc(file=sys.stdout)
                    pass
            #cv2.imshow('original',self.frame)
            self.show()
        self.close()
    def getTrianglePose(self,points):
        #Get the angle and position (pose) of the robot
    
        #get closest points (wich form the base triangle)
        dist01=np.sqrt(np.power(points[0][0]-points[1][0],2)+np.power(points[0][1]-points[1][1],2))
        dist12=np.sqrt(np.power(points[1][0]-points[2][0],2)+np.power(points[1][1]-points[2][1],2))
        dist20=np.sqrt(np.power(points[2][0]-points[0][0],2)+np.power(points[2][1]-points[0][1],2))
        if dist01<dist12 and dist01<dist20:
            midx=(points[0][0]+points[1][0])/2
            midy=(points[0][1]+points[1][1])/2
            arrow=points[2]
        elif dist12<dist01 and dist12<dist20:
            midx=(points[1][0]+points[2][0])/2
            midy=(points[1][1]+points[2][1])/2
            arrow=points[0]
        elif dist20<dist01 and dist20<dist12:
            midx=(points[2][0]+points[0][0])/2
            midy=(points[2][1]+points[0][1])/2
            arrow=points[1]
        cx=midx
        cy=midy
        
        cv2.arrowedLine(self.frame,(cx,cy),(arrow[0],arrow[1]),(255,0,0),2)
        cv2.circle(self.frame,(cx,cy),1,(60,255,255),1,8,0)
        arrow2=list()
        arrow2.append(float(arrow[0]-cx))    
        arrow2.append(float(cy-arrow[1])) 
        m=np.arctan2(float((arrow2[1])),float(arrow2[0]))
        #to global coordinate frame
        m=np.degrees(m)
        m=round(m,2)
        cv2.putText(self.frame, '('+str(cx)+','+str(cy)+')',(cx-10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255))
        cv2.putText(self.frame, str(m),(cx-20,cy-20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255))
        self.angle = m
        self.pos = (cx,cy)
    def show(self):
        # Show the images
        cv2.imshow('mask', self.mask)
        cv2.imshow('Camara', self.frame)
        tecla = cv2.waitKey(5) & 0xFF
        if tecla == 27:
            self.continue_=False
    def close(self):
        '''
        Close all windows created by the main object        
        '''
        self.sender.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cT = ChotobotTracker(30,60,255)
    cT.start()
    try:
        cT.process()
    except:
        traceback.print_exc(file=sys.stdout)
        cT.close()
