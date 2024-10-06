# For video streaming:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html

# For RGB/HSV thresholding:
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
	# (More complex ex (uses sliders to dynamically adjust the threshold), not used in this program):
	# https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html

# For dominant color w/ K-means clustering:
# https://github.com/opencv/opencv/blob/master/samples/python/kmeans.py
# https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
# The dominant color is determined over a medium-sized rectangle at the center of the frame.
# This program also prints out the dominant BGR value to console.


'''
1. (tracking.jpeg)
	BGR and HSV can both fairly accurately track an object through consistent lighting conditions,
	but BGR tends to have more false positives(random white pixels), so HSV is overall more precise.
	The bounds on hue are very narrow, and it is much more intuitive.

2. Changing the lighting conditions makes it difficult to track the item for a specific BGR/HSV threshold,
	especially since my iphone camera is trying to adjust to the colors.

3. (screenBrightness.jpeg)
	For BGR (222, 184, 16), a shade of sky blue, the dominant color observed is about the same, although
	less saturated (B & G identical, but R larger than actual).
	There is no difference in percieved color between a medium and very bright screen.

4. (bookBackground.jpeg vs screenBackground.jpeg)
	For changing the lighting conditions around a object vs a screen, the dominant color
	of the object changes very drastically, especially since I used a warm-colored light. I can see
	this tendency becoming very troublesome in an actual project.
'''

import cv2 as cv
import sys
import numpy as np


def trackingBox():
	cv.namedWindow("rectangle", cv.WINDOW_NORMAL)
	cap = cv.VideoCapture(0)
	# 0 for default camera(synced to iphone (settings->continuity camera), then switched to webcam after disconnect)
	while(True):
		valid, frame = cap.read()
		if valid:
			# BGR: 90, 95, 90
			# thresholdFrameBGR = cv.inRange(frame, (220, 140, 0), (255, 170, 60))
			thresholdFrameHSV = cv.inRange(cv.cvtColor(frame, cv.COLOR_BGR2HSV), (101, 195, 215), (103, 255, 255))


			# draw rectangle around all non-zero pixels in grayscale image
			x,y,w,h = cv.boundingRect(thresholdFrameHSV)	#or use the BGR threshold
			cv.rectangle(frame, (x,y), (x+w, y+h), 255, 5)
			cv.imshow("rectangle", frame)

		else:
			print("Invalid frame")
		if cv.waitKey(1) == 27: # ESC
			break
	cap.release()


# Find the dominant color of the pixels at the center of the screen
def dominantColor():
	cv.namedWindow("frame", cv.WINDOW_NORMAL)
	cv.namedWindow("dominant", cv.WINDOW_NORMAL)
	cap = cv.VideoCapture(0)
	while(True):
		valid, frame = cap.read()
		if valid:
			height = int(len(frame)/3)
			width = int(len(frame[0])/3)
			centerFrame = frame[height:2*height, width:2*width]
			# print(len(frame))
			# print(len(centerFrame))
			# reshape center of frame to 2D and convert to floats
			centerFrame = np.float32(centerFrame.reshape(-1,3))
			# 1 cluster (i.e. 1 color)
			# try 10 iterations
			criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			_, labels, centers = cv.kmeans(centerFrame, 1, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

			# get the dominant color
			domColor = list(map(int, centers[0]))
			print(domColor)
			centerFrame = np.uint8(centerFrame.reshape(height, width, 3))

			# draw rectangle of part of frame used to find dominant color
			cv.rectangle(frame, (width, height), (2*width, 2*height), 255, 5)
			# draw filled in rectangle of dominant color
			cv.rectangle(centerFrame, (0,0), (width, height), domColor, -1)

			cv.imshow("frame", frame)
			cv.imshow("dominant", centerFrame)
		else:
			print("Invalid frame")
		if cv.waitKey(1) == 27:
			break
	cap.release()


if __name__ == '__main__':
	trackingBox()
	# print(cv.cvtColor(np.uint8([[[220,140,0]]]),cv.COLOR_BGR2HSV)[0][0])
	# print(cv.cvtColor(np.uint8([[[255,170,60]]]),cv.COLOR_BGR2HSV)[0][0])
	# dominantColor()

