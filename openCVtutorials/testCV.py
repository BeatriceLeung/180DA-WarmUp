# from open CV tutorials

import cv2 as cv
import sys
import numpy as np



def displayPicture(img):
	cv.imshow("display window", img)
	k = cv.waitKey(0) # waits infinitely for key event

	# if k==ord("s"): #ord gets its unicode code point
	# 	cv.imwrite("picture.jpeg", img) #overwrites the file

	# works, altho image is veeeery big if not resized

#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
#Change the color scheme + use a range of colors as a mask
def changeColorSpace(img):
	windows = ["hsv colors","gray colors", "green mask", "red mask", "red and green mask"]
	for name in windows:
		cv.namedWindow(name, cv.WINDOW_NORMAL) #resizable window

	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	cv.imshow("hsv colors", hsv)
	cv.imshow("gray colors", gray)

	maskgreen = cv.inRange(img, np.array([0,20,5]), np.array([55,235,200]))
	maskred = cv.inRange(img, np.array([56, 58, 91]), np.array([160,120,250]))
	green = cv.bitwise_and(img, img, mask = maskgreen)
	red = cv.bitwise_and(img, img, mask = maskred)
	greenAndRed = cv.bitwise_or(green, red)
	cv.imshow("green mask", green)
	cv.imshow("red mask", red)
	cv.imshow("red and green mask", greenAndRed)

	k = cv.waitKey(0) # waits infinitely for key event
	## cv.inRange(imageArray, color1, color2) to get mask
	## cv.bitwise_and(imageArray, ??) to apply mask


#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#For grayscale only?
#Filtering based on color (thresholding, set to 0 or max if pixel below/above threshold)
#There's also "adaptive thresholding," which will take each pixel's surroundings into account
#Good for shadows/lighting
#And there's some method to automatically choose the threshold (Otsu)
def thresholding(img):
	windows = ["gray colors","simple threshold", "adaptive threshold"]
	for name in windows:
		cv.namedWindow(name, cv.WINDOW_NORMAL) #resizable window

	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	_ , simpleBinary = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)
	adaptiveGaussian = cv.adaptiveThreshold(cv.medianBlur(gray, 5), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
		cv.THRESH_BINARY, 11, 2) # how many pixels to use; constant to subtract from mean

	cv.imshow("gray colors", gray)
	cv.imshow("simple threshold", simpleBinary)
	cv.imshow("adaptive threshold", adaptiveGaussian) #with some smoothening, tho not very useful here

	k = cv.waitKey(0) # waits infinitely for key event


#Try edge detection later w/ cv.Canny(image, threshold1, threshold2) using gradients

#https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
#Slide template across base image and find matches
#Result slightly smaller than base image b/c template stays within image bounds
#each pixel in result image is match result for template with top-left coordinates at that point
# ** Template image needs to be rescaled to be big enough!
def templateMatch(img, templateImg):
	cv.namedWindow("template match", cv.WINDOW_NORMAL)
	cv.namedWindow("with rectangle", cv.WINDOW_NORMAL)
	grayTemplateImg = cv.cvtColor(templateImg, cv.COLOR_BGR2GRAY)
	matches = cv.matchTemplate(cv.cvtColor(img, cv.COLOR_BGR2GRAY), grayTemplateImg , \
		cv.TM_CCOEFF) #there are other comparison methods
	_, _, _, top_left = cv.minMaxLoc(matches)  # get position of max value

	bottom_right = (top_left[0]+grayTemplateImg.shape[::-1][0], top_left[1]+grayTemplateImg.shape[::-1][1])
	matchRectangle = cv.rectangle(img, top_left, bottom_right, 255, 5)
	cv.imshow("template match", matches)
	cv.imshow("with rectangle", matchRectangle)
	cv.waitKey(0)

#https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
#https://docs.opencv.org/4.x/d7/d1d/tutorial_hull.html
#Various moments/contour stuff
#Convex hull makes convex contours bulge out instead -- skipped
def contours(img):
	cv.namedWindow("threshold", cv.WINDOW_NORMAL)
	cv.namedWindow("rectangle", cv.WINDOW_NORMAL)
	# cv.namedWindow("convex hull", cv.WINDOW_NORMAL)
	_, threshold = cv.threshold(cv.medianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 35), 220, 255, cv.THRESH_BINARY)
	# contours, _ = cv.findContours(threshold, 1, 2) #?
	# hull = cv.convexHull(contours[0])
	# then plot the contour line

	x,y,w,h = cv.boundingRect(threshold) #finds rectangle around all non-zero pixels in grayscale image
	cv.imshow("threshold", threshold)
	cv.rectangle(threshold, (x, y), (x+w, y+h), 255, 5)
	cv.imshow("rectangle", threshold)

	cv.waitKey(0)

#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
#May be useful for detecting objects at different scales (HOG descriptor)
#https://github.com/opencv/opencv/blob/master/samples/python/peopledetect.py
#Below code follows the object with a bounding rectangle, but struggles with different distances/angles
def playWebcam(trackImg):
	# cv.namedWindow("camera", cv.WINDOW_NORMAL)
	cap = cv.VideoCapture(0)
	#0 for default camera(synced to iphone (settings->continuity camera), then switched to webcam after disconnect)
	while(True):
		valid, frame = cap.read()
		if valid:
			print("Valid frame")
			matches = cv.matchTemplate(frame, trackImg, cv.TM_CCOEFF) #there are other comparison methods
			_, _, _, top_left = cv.minMaxLoc(matches)  # get position of max value
			bottom_right = (top_left[0]+trackImg.shape[:-1][0], top_left[1]+trackImg.shape[:-1][1])
			cv.rectangle(frame, top_left, bottom_right, 255, 5)
			cv.imshow("frame", frame)
		else:
			print("Invalid")
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()


if __name__ == '__main__':
	img = cv.imread("picture.jpeg")
	img2 = cv.imread("picture2.jpeg")
	templateImg = cv.imread("house.jpg")
	trackImg = cv.imread("realObject.jpeg")
	if img is None or templateImg is None:
		sys.exit("Couldn't read file(s)")
	# displayPicture(img)
	# changeColorSpace(img)
	# thresholding(img)
	# templateMatch(img2, templateImg)
	# contours(img2)
	playWebcam(trackImg)
	cv.destroyAllWindows()


