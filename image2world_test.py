import cv2 as cv
import numpy as np

def invert(uvPoints, rvecinv, tvec, inverse_camera, src):
    point_coor = []
    xy_coor = []
    for index, i in enumerate(uvPoints):
            i.append([1]) # add fake z coordinate   
            uvPoint = i
            
            tempMat = rvecinv * inverse_camera * uvPoint
            tempMat2 = rvecinv * tvec
            print('t1', tempMat, tempMat[2,0])
            print('t2', tempMat2, tempMat2[2,0])
            s = tempMat2[2,0]
            s /= tempMat[2,0] 
           
            translated = s*inverse_camera * uvPoint  - tvec
            
            xyz = rvecinv*translated
            x = xyz[0][0]
            y = xyz[1][0]
            
            if index == 0:
                offset_x = x
                offset_y = y 
            x -= offset_x
            y -= offset_y
            if index == 1:
                yscale = y/17
            if index == 3:
                xscale = x/24  
            if index == 4:
                xscale = 0.5*(xscale + x/24)
                yscale = 0.5*(yscale + y/17)

            point_coor.append((x,y))  
            print('\nimage point', uvPoint)
    for index,i in enumerate(point_coor):
            x = i[0]/xscale
            y = i[1]/yscale
            xy_coor.append((x,y))
            out = 'x: {}, y: {}'.format(round(x, 2),round(y, 2))
            print('world coords', out ) 
            cv.putText(src, out, (uvPoints[index][0][0], uvPoints[index][1][0]), cv.FONT_HERSHEY_SIMPLEX,  .5, (0,255,0), 2)
    return src, xy_coor

def uv_to_xy(img, params, uvPoints, debug = False):
    src = img
    ret, mtx, dist, rvecs, tvecs = params
    '''Hardcoded to calibration image'''
    #objectPoints = [[[0],[23.1], [0]], [[26.0],[22.9], [0]], [ [26.5],[0.7], [0]], [[0], [0], [0]], [[13.5], [11.8], [0]] ]
    objectPoints = [[[0],[17], [0]], [[24],[17], [0]], [ [24],[0], [0]], [[0], [0], [0]], [[12], [9], [0]] ]
    imagePoints = []
    
    '''Locating keypoints'''
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=20, maxRadius=50)

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
    print("obj points", objectPoints, "imagePoints",  imagePoints, "mtx", mtx)
    ret, rvec, tvec, fatt = cv.solvePnPRansac(objectPoints, imagePoints, mtx, distCoeffs=dist, flags = cv.SOLVEPNP_ITERATIVE) 
    print("rvec", rvec)
    rotationMat = cv.Rodrigues(rvec)[0]
    rvec = rvec.T
    rotationMatInverse = np.linalg.inv(rotationMat)
    rvecinv = cv.Rodrigues(rotationMatInverse)[0]

    inverse_camera = np.linalg.inv(mtx)
    inverse_camera = cv.Rodrigues(inverse_camera)[0].T

    tvec = tvec.T

    print('--------------MATRICES--------------')
    print('rvecinv', rvecinv.shape)
    print('tvec', tvec.shape)
    print('intrinsic camera params', mtx)
    print('inverse camera mtx', inverse_camera.shape)
    print()

    if debug:
        return invert(imagePointsCopy, rvecinv, tvec, inverse_camera, src)
    else:
        return invert(uvPoints, rvecinv, tvec, inverse_camera, src)
