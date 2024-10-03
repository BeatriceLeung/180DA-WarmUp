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
		cv.THRESH_BINARY, 11, 2)
	# how many pixels to use; constant to subtract from mean

	cv.imshow("gray colors", gray)
	cv.imshow("simple threshold", simpleBinary)
	cv.imshow("adaptive threshold", adaptiveGaussian) #with some smoothening, tho not very useful here

	k = cv.waitKey(0) # waits infinitely for key event


#Try edge detection later w/ cv.Canny(image, threshold1, threshold2) using gradients




if __name__ == '__main__':
	img = cv.imread("picture.jpeg")
	if img is None:
		sys.exit("Couldn't read file")
	# displayPicture(img)
	# changeColorSpace(img)
	# thresholding(img)
	cv.destroyAllWindows()


