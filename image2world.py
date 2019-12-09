import cv2 as cv
import numpy as np

class ImageToWorld:
	def __init__(self, cameramtx, dist, objectPoints):
		'''ex objectPoints [[[25.1],[3.1], [0]], [[25.8],[19], [0]], [ [4.3],[18.1], [0]], [[4], [3.5], [0]] ]'''
		self.mtx = cameramtx
		self.dist = dist
		self.objectPoints = np.float32(sorted(objectPoints, key = lambda pointSum: pointSum[0][0] + pointSum[1][0]))
		self.imagePoints = []


	def locate_keypoints(self, src):

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
		        self.imagePoints.append([[i[0]], [i[1]]])

		self.imagePoints.sort(key = lambda pointSum: pointSum[0][0] + pointSum[1][0])

		print('--------------KEYPOINTS--------------')
		print(self.objectPoints)
		print(self.imagePoints)

		self.imagePoints = np.float32(self.imagePoints)

	def find_inverse_params(self):
		'''Finding inverse for pinhole camera equation: [x y z] = R [X Y Z] + t'''
		if self.imagePoints == []:
			print('Locate keypoints first')
			return

		ret, rvec, tvec, inliers = cv.solvePnPRansac(self.objectPoints, self.imagePoints, self.mtx, distCoeffs=self.dist, flags = cv.SOLVEPNP_P3P)#cv.SOLVEPNP_ITERATIVE) #
		rotationMat = cv.Rodrigues(rvec)[0]
		rvec = rvec.T
		rotationMatInverse = np.linalg.inv(rotationMat)
		self.rvecinv = cv.Rodrigues(rotationMatInverse)[0]

		inverse_camera = np.linalg.inv(mtx)
		self.inverse_camera = cv.Rodrigues(inverse_camera)[0].T

		self.tvec = tvec.T

		print('--------------MATRICES--------------')
		print('rvecinv', rvecinv.shape)
		print('tvec', tvec.shape)
		print('intrinsic camera params', mtx)
		print('inverse camera mtx', inverse_camera.shape)
		print()

	def convert_to_xy(self, imgPoints, src):
		ret = []
		for index, i in enumerate(imgPoints):
			i.append([1]) # add fake z coordinate   
			uvPoint = i
			
			tempMat = self.rvecinv * self.inverse_camera * uvPoint
			tempMat2 = self.rvecinv * self.tvec
			print('t1', tempMat, tempMat[2,0])
			print('t2', tempMat2, tempMat2[2,0])
			s = tempMat2[2,0]
			s /= tempMat[2,0] *3 # TODO fix scaling param calulation, should be div by number of calibration points
			print('scaling factor', s)
			
			translated = s* self.inverse_camera * uvPoint  - self.tvec
			xyz = self.rvecinv * translated

			x = xyz[0][0]
			y = xyz[1][0]
			if index == 0:
			    offset_x = -x
			    offset_y = y
			x =-x - offset_x
			y -= offset_y  

			print('\nimage point', uvPoint)
			ret.append([[x],[y]])
			out = 'x: {}, y: {}'.format(round(x, 2),round(y, 2))
			print('world coords', out ) 
			cv.putText(src, out, (i[0][0], i[1][0]), cv.FONT_HERSHEY_SIMPLEX,  .5, (0,255,0), 2)

		return ret, src
