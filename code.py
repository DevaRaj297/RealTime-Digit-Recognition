import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import time
import sys

clf = joblib.load("num_classifier.model");

def processFrame(img):
	digits='Predicted digits are  ';
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
	img_gray = cv2.GaussianBlur(img_gray, (3,3), 0);
	ret, img_th = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)# Threshold the image
	_, ctrs, hier = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);# Find contours
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	rects.sort();
	for rect in rects:
		if(rect[0]>x1 and rect[1]>y1 and rect[0]<(x1+wid) and rect[1]<(y1+hig) and (rect[0] + rect[2])<(x1+wid) and (rect[1] + rect[3])<(y1+hig)):
			cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 1);
			x=rect[0]-val;y=rect[1]-val;
			w=rect[2]+(2*val);h=rect[3]+(2*val);
			
			roi = img_th[y:y+h, x:x+w];
			roi = cv2.resize(roi, (28, 28));
			roi = cv2.dilate(roi, (3,3));
			roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=None, block_norm = 'L1')

			nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
			cv2.putText(img, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2);
			digits  = digits+str(int(nbr[0]));
	cv2.putText(img, digits, (30, 330),cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 250, 50), 1);
	cv2.imshow("Digit Recognition",img);
	k = cv2.waitKey(0) & 0xff
	if k == ord('q'):
		stop();
				
def stop():
	cam.release();
	cv2.destroyAllWindows();
	sys.exit(0);

if __name__ == '__main__':
	x1=100;y1=100;wid=440;hig=150;val = 20;
	string1='Press "p" to predict digits in the frame ';
	string2='Press "q" to QUIT the programe ';
	
	cam = cv2.VideoCapture(0);
	while(True):
		(ret, frame) = cam.read();
		frame = cv2.flip(frame, 1);
		cv2.rectangle(frame, (x1, y1), (x1+wid, y1+hig), (255, 255, 255), 2);
		cv2.putText(frame, string1, (20, 20),cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1);
		cv2.putText(frame, string2, (20, 50),cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1);
		cv2.imshow("Digit Recognition",frame);
		k = cv2.waitKey(1) & 0xff
		if k == ord('p'):
			processFrame(frame);
		elif k == ord('q'):
			stop();

