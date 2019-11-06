import cv2 as cv


def show_webcam(mirror=False):
    cam = cv.VideoCapture(0)

    '''
    cv.VideoCapture.set(cam.CV_CAP_PROP_FRAME_HEIGHT, 320)
    cv.VideoCapture.set(cam.CV_CAP_PROP_FRAME_WIDTH, 240)
    '''
    cam.set(3,720)
    cam.set(4,720)
    tess = cv.text.OCRTesseract_create()
    tCNN = cv.text.TextDetectorCNN_create('textbox.prototxt', 'TextBoxes_icdar13.caffemodel')
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv.flip(img, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY )
        #cv.threshold(src=img, thresh=100, maxval=255, type=0, dst=img)
        img = cv.adaptiveThreshold(src=img, maxValue=255, adaptiveMethod=1, thresholdType=0, blockSize=7, C=14)
        #img = cv.Canny(img, 50, 200, None, 3)
        #print(cv.text_OCRTesseract.run(img))

        '''
        Bbox, confidence = tCNN.detect(img)

        for i in range(len(confidence)):
            if confidence[i] > .03:
                cv.rectangle(img,(Bbox[i][0],Bbox[i][1]),(Bbox[i][0] + Bbox[i][2], Bbox[i][1] + Bbox[i][3]),(0,255,0),3)
        '''
        cv.imshow('output', img)
        if cv.waitKey(1) == 27: 
            break  # esc to quit
    cv.destroyAllWindows()


def main():
    show_webcam()


if __name__ == '__main__':
    main()
