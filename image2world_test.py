import pipeline as p
import cv2 as cv

cam = cv.VideoCapture(0, cv.CAP_V4L2)
cam.set(3, 1280)  # height
cam.set(4, 720)  # width

img = p.capture_image(cam)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


canny = cv.Canny(gray, 50, 150, None, 3)
contours,hier = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

if len(contours) == 0:
    print("Didn't find any contours!")
    

for i, cnt in enumerate(contours):
    if cv.contourArea(cnt)>500:  # only grab large contours
        print('contour:', i)
        #cv.drawContours(img,[cnt],0,(0,255,0),2)
        hull = cv.convexHull(cnt) # find the convex hull of contour
        hull = cv.approxPolyDP(hull,0.1*cv.arcLength(hull,True),True)
        cv.drawContours(img,[hull],0,(0,255,0),2)

    

p.display(img)
