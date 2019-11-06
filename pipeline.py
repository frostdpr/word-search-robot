import cv2 as cv
import pytesseract
import numpy as np
import math 

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract' # windows only


def capture_image(device, mirror=False, debug=False):
    
    while True:
        ret_val, img = device.read()
        if mirror: 
            img = cv.flip(img, 1)

        if debug:
            return img
        cv.imshow('output', img)

        if cv.waitKey(1) == 27: 
            cv.destroyAllWindows()
            return img

def display(img):
    while True:
        cv.imshow('img', img)
        if cv.waitKey(1) == 27: 
            cv.destroyAllWindows()
            break;

def segment(img, prob=False, debug=False):
    '''
    dst = cv.cvtColor(img, cv.COLOR_BGR2GRAY )
    
    #kernel = np.ones((1,1),np.uint8)
    #dst = cv.erode(dst,kernel,iterations = 20)
    dst = cv.Canny(dst, 50, 150, None, 3)
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    if prob: #use probabilistic hough transform
         linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 200, None, 300, 50)
    
         if linesP is not None:
             for i in range(0, len(linesP)):
                 l = linesP[i][0]
                 cv.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
         print(linesP)
    else:
        lines = cv.HoughLines(dst, 1, np.pi / 180, 200  , None, 0, 0)
        
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

            print(lines)

    while True and not debug:
        cv.imshow('output', cdst)
        if cv.waitKey(1) == 27: 
            cv.destroyAllWindows()
            break;
    return cdst
    #return (puzzle_main, puzzle_bank)
    '''
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255 

    #ret,thresh = cv.threshold(gray,127,255,0)
    thresh = cv.Canny(gray, 50, 150, None, 3)
    contours,hier = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv.contourArea(cnt)>5000:  #remove small areas like noise etc
            hull = cv.convexHull(cnt) #find the convex hull of contour
            hull = cv.approxPolyDP(hull,0.1*cv.arcLength(hull,True),True)
            #print(hull)
            if len(hull)==4:
                #cv.drawContours(img,[hull],0,(0,255,0),2)
                x,y,w,h = cv.boundingRect(cnt) # get minimal bounding for contour
                cv.drawContours(mask, [cnt], -1, 0, -1)
                break

    puzzle = img[y:y+h,x:x+w]
    bank = cv.bitwise_and(img, img, mask=mask)

    while True and not debug:
        cv.imshow('output', bank)
        if cv.waitKey(1) == 27: 
            cv.destroyAllWindows()
            break;
    return puzzle, bank
 
def tesseract(puzzle, bank):
    puzzle = cv.resize(puzzle, (0,0), fx=3, fy=3)
    bank = cv.resize(bank, (0,0), fx=1.5, fy=1.5)
    cv.threshold(src=puzzle, thresh=100, maxval=255, type=0, dst=puzzle)
    cv.threshold(src=bank, thresh=95, maxval=255, type=0, dst=bank)
    puzzle = cv.rotate(puzzle, cv.ROTATE_90_CLOCKWISE)
    bank = cv.rotate(bank, cv.ROTATE_90_CLOCKWISE)
    display(puzzle)
    display(bank)
    #img = cv.adaptiveThreshold(src=img, maxValue=255, adaptiveMethod=1, thresholdType=0, blockSize=7, C=14)
    puzzle_config = r'--tessdata-dir ./protos_data -l eng  --oem 0 --psm 3'
    bank_config = r'--oem 3, psm 3'
    puzzle_detection = pytesseract.image_to_string(puzzle, config = puzzle_config)
    bank_detection = pytesseract.image_to_string(bank)
    parsed_puzzle_detection = [[i.split(' ')] for i in puzzle_detection.split('\n')]
    rotated_puzzle = list(zip(*parsed_puzzle_detection[::-1]))
    print('-------------------PUZZLE----------------------')
    #print('/n'.join(rotated_puzzle[0]))
    clean_puzzle = rotated_puzzle[0]
    for j in range(len(clean_puzzle[0][0])):		
        for idx, item in enumerate(clean_puzzle):
            try:
                print(clean_puzzle[idx][0][j], end='')
            except:
                print('*', end='')
        print('')

            
    #print(puzzle_detection)
    print('-------------------WORD BANK----------------------')
    print(bank_detection)

def debug(function, device):

    while True:
        img = capture_image(device, debug=True)
        output  = function(img, debug=True)
        cv.imshow('output', output)
        if cv.waitKey(1) == 27: 
            break;

    cv.destroyAllWindows()

def main():
    cam = cv.VideoCapture(0)
    cam.set(3,1280) #height
    cam.set(4,720) #width

    img = capture_image(cam)

    puzzle, bank = segment(img)
    tesseract(puzzle, bank)

if __name__ == '__main__':
    main()
