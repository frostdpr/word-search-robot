import cv2 as cv
import draw as drw
import pipeline as p
import image2world as i2w
jetson_UART = "/dev/ttyTHS1"
drawer = drw.Drawer()


start_offset = 13
try:
    cam = cv.VideoCapture(0, cv.CAP_V4L2)
    cam.set(3, 1280)  # height
    cam.set(4, 720)  # width
    
    objPoints = [[[26.65],[1], [0]], [[26.05],[22.2], [0]], [ [0],[23], [0]], [[0], [0], [0]] ]
    params = p.chessboard_calibrate('calibration', 6, 8, debug=False)
    ret, mtx, dist, rvecs, tvecs = params
    
    img = p.capture_image(cam) # calibration first
    xyz = i2w.ImageToWorld(mtx, dist, objPoints)
    xyz.locate_keypoints(img)
    
    xyz.find_inverse_params()
    img = p.capture_image(cam) # puzzle
    
    drawPoints, src = xyz.convert_to_xy(getattr(xyz, 'imagePoints'), img)
    
    p.display(src)
    
    x1 = int(drawPoints[0][0]/.225) + start_offset
    y1 = int(drawPoints[0][1]/.225) + start_offset
    x2 = int(drawPoints[1][0]/.225) + start_offset
    y2 = int(drawPoints[1][1]/.225) + start_offset
    
    test = [(x1, y1), (x2, y2)]
    print(test)
    #test = [(13,13), (116+start_offset,99+start_offset)]
    drawer.draw(test)
    

except Exception as e:
    print(str(e))
    drawer.cleanup()
