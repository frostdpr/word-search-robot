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
'''
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
img = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
src = img[y:y+h, x:x+w]
'''
#p.display(img, 'Calibration Output')

src = img
'''Hardcoded to calibration image'''
objectPoints = [[[25.1],[3.1], [0]], [[25.8],[19], [0]], [ [4.3],[18.1], [0]], [[4], [3.5], [0]] ]
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
        cv.circle(src, center, 1, (0, index*50 , 255), 3)
        imagePoints.append([[i[0]], [i[1]]])



objectPoints.sort(key = lambda pointSum: pointSum[0][0] + pointSum[1][0])
imagePoints.sort(key = lambda pointSum: pointSum[0][0] + pointSum[1][0])

print('--------------KEYPOINTS--------------')
print(objectPoints)
print(imagePoints)


objectPoints = np.float32(objectPoints)
imagePointsCopy = imagePoints.copy()
imagePoints = np.float32(imagePoints)



'''Finding inverse for pinhole camera equation: [x y z] = R [X Y Z] + t'''
ret, rvec, tvec, inliers = cv.solvePnPRansac(objectPoints, imagePoints, mtx, distCoeffs=dist, flags = cv.SOLVEPNP_P3P)#cv.SOLVEPNP_ITERATIVE) #
rotationMat = cv.Rodrigues(rvec)[0]
rvec = rvec.T
rotationMatInverse = np.linalg.inv(rotationMat)
rvecinv = cv.Rodrigues(rotationMatInverse)[0]

inverse_camera = np.linalg.inv(mtx)
inverse_camera = cv.Rodrigues(inverse_camera)[0].T
imagePointsCopy.append([[690],[380]]) #roughly center of paper

tvec = tvec.T

print('--------------MATRICES--------------')
print('rvecinv', rvecinv.shape)
print('tvec', tvec.shape)
print('intrinsic camera params', mtx)
print('inverse camera mtx', inverse_camera.shape)
print()


for index, i in enumerate(imagePointsCopy):
	i.append([1]) # add fake z coordinate   
	uvPoint = i
	
	tempMat = rvecinv * inverse_camera * uvPoint
	tempMat2 = rvecinv * tvec
	print('t1', tempMat, tempMat[2,0])
	print('t2', tempMat2, tempMat2[2,0])
	s = tempMat2[2,0]
	s /= tempMat[2,0] *3 # TODO fix scaling param calulation, should be div by number of calibration points
	print('scaling factor', s)
	
	translated = s*inverse_camera * uvPoint  - tvec
	#translated = uvPoint - tvec
	#print('t',translated)
	xyz = rvecinv*translated
	x = xyz[0][0]
	y = xyz[1][0]
	if index == 0:
	    offset_x = -x
	    offset_y = y
	x =-x - offset_x
	y -= offset_y  

		
	print('\nimage point', uvPoint)

	out = 'x: {}, y: {}'.format(round(x, 2),round(y, 2))
	print('world coords', out ) 
	cv.putText(src, out, (i[0][0], i[1][0]), cv.FONT_HERSHEY_SIMPLEX,  .5, (0,255,0), 2)
p.display(src)
