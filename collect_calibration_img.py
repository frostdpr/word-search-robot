import cv2 as cv
import os
import random
import string

if not os.path.exists("calibration"):
    os.makedirs("calibration")

cam = cv.VideoCapture(0)
cam.set(3, 1280)  # height
cam.set(4, 720)  # width
letters = string.ascii_lowercase

while True:
    ret_val, img = cam.read()

    cv.imshow("output", img)
    key = cv.waitKey(1)
    if key == 27:  # esc
        cv.destroyAllWindows()
        break
    elif key == 32:  # space
        cv.destroyAllWindows()
        cv.imwrite(
            "calibration/" + "".join(random.choice(letters) for i in range(6)) + ".png",
            img,
        )
