import pipeline as p
import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0, cv.CAP_V4L2)
cam.set(3, 1280)  # height
cam.set(4, 720)  # width

img = cv.imread('test_searches/pmmfxv.png')
#img = p.capture_image(cam)
params = p.chessboard_calibrate('calibration', 6, 8, debug=False)
ret, mtx, dist, rvecs, tvecs = params
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
img = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
src = img[y:y+h, x:x+w]

#p.display(img, 'Calibration Output')


'''Hardcoded to calibration image'''
objectPoints = [[[25.1],[3.1], [0]], [[25.8],[19],   [0]], [ [4.3],[18.1], [0]], [[4], [3.5], [0]] ]
imagePoints = []


'''Locating keypoints'''
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)
rows = gray.shape[0]
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                           param1=100, param2=30,
                           minRadius=1, maxRadius=30)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for index, i in enumerate(circles[0, :]):
        center = (i[0], i[1])
        cv.circle(src, center, 1, (0, 255, index*50), 3)
        imagePoints.append([[i[0]], [i[1]]])

objectPoints = np.float32(objectPoints)
imagePointsCopy = imagePoints.copy()
imagePoints = np.float32(imagePoints)

'''Finding inverse for pinhole camera equation: [x y z] = R [X Y Z] + t'''
ret, rvec, tvec, inliers = cv.solvePnPRansac(objectPoints, imagePoints, newcameramtx, distCoeffs=None, flags = cv.SOLVEPNP_P3P)

rotationMat = cv.Rodrigues(rvec)
print(len(rotationMat))

rotationMatInverse = np.linalg.inv(rotationMat[0])

rvecinv = cv.Rodrigues(rotationMatInverse)[0]

imagePointsCopy.append([[690],[380]]) #roughly center of paper

for i in imagePointsCopy:
	i.append([0]) # add fake z coordinate
	uvPoint = i
	print(uvPoint)
	translated = uvPoint - tvec
	#print('t',translated)
	xyz = rvecinv*translated
	print('x', xyz[0][0]*-1-13, 'y', xyz[1][0] - 4.7) #TODO fix origin offset and inverted x coord

#p.display(img)
