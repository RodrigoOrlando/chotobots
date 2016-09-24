import cv2
import numpy as np
import traceback,sys
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt

class ChotobotTracker(object):
	def __init__(self):
		self.cap=None
		self.minArea = 1000
		self.maxArea = 4000
		self.eps = 10
		self.th = 100
		self.maxth = 255

		cv2.namedWindow("src",cv2.WINDOW_NORMAL)
		cv2.createTrackbar("minArea","src",self.minArea,100000,self.nothing)
		cv2.createTrackbar("maxArea","src",self.maxArea,100000,self.nothing)
		cv2.createTrackbar("eps","src",self.eps,100,self.nothing);
		cv2.createTrackbar("th","src",self.th,255,self.nothing);
		cv2.createTrackbar("maxth","src",self.maxth,255,self.nothing);

	def start(self):
		#self.cap = cv2.VideoCapture('video.mp4')
		self.cap = cv2.VideoCapture(0)
		self.continue_=True
	def close(self):
	 	cv2.destroyAllWindows()
	def nothing(self,x):
		pass
	def drawAxis(self,img,p,q,colour,scale=0.2):
		angle = np.arctan2([p[1]-q[0][1]],[p[0]-q[0][0]])
		degrees = angle * 180 /np.pi
		cv2.arrowedLine(img,p,tuple(q[0]),colour,scale)
		cv2.putText(img, str(p),(int(p[0])-10,int(p[1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,255))
		degrees=abs(degrees-180)
		cv2.putText(img, str(degrees),(int(p[0])-20,int(p[1])-20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,255))
	def getOrientation(self, pts, img, corner):
		sz=pts.shape
		data_pts = np.zeros((sz[0],sz[2],1), dtype=np.float64)
		data_pts[:,:,0] = pts[:,0,:] 
		#print(data_pts,type(data_pts))
		mean, eigenvectors = cv2.PCACompute(data_pts)
		cntr=(int(mean[0][0]),int(mean[0][1]))
		cv2.circle(img, cntr, 3, (255, 0, 255), 2);
		self.drawAxis(img,cntr,corner,(0, 255, 0),1)
		#print(mean, type(mean))
	def process(self):
		corner=None
		while(self.continue_):
		 	_,src = self.cap.read()
			self.minArea=cv2.getTrackbarPos("minArea","src")
			self.maxArea=cv2.getTrackbarPos("maxArea","src")
			self.eps=cv2.getTrackbarPos("eps","src")
			self.th=cv2.getTrackbarPos("th","src")
			self.maxth=cv2.getTrackbarPos("maxth","src")
			gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
			ret,bw = cv2.threshold(gray, self.th, self.maxth, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			cv2.imshow("bw",bw)
			contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
			aprox_con=list()
			for i in contours:
				area=cv2.contourArea(i)
				if (area < self.minArea or self.maxArea < area): continue
				aprox_pts=cv2.approxPolyDP(i,self.eps,True)
				sz=len(aprox_pts)
				if (sz==3):
					aprox_con.append(i)
					a = cv2.norm(aprox_pts[0]-aprox_pts[1])
					b = cv2.norm(aprox_pts[0]-aprox_pts[2])
					c = cv2.norm(aprox_pts[1]-aprox_pts[2])
					if a<b and a<c: 
						corner = aprox_pts[2]
					elif b<a and b<c: 
						corner = aprox_pts[1]
					elif c<a and c<b: 
						corner = aprox_pts[0]
					else: 
						corner=None
			        if corner!=None:
			        	self.getOrientation(i, src, corner);
					cv2.drawContours(src, aprox_con, -1, (0, 0, 255), 2, 8);
	 		
	 		cv2.imshow("SRC",src)
			tecla = cv2.waitKey(5) & 0xFF
			if tecla == 27:
				self.continue_=False
	 	self.close()

if __name__ == '__main__':
    cT = ChotobotTracker()
    cT.start()
    try:
        cT.process()
    except:
        traceback.print_exc(file=sys.stdout)
        cT.close()
