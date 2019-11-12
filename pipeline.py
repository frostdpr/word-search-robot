#!/usr/bin/env python3

import cv2 as cv
import pytesseract
import numpy as np
import math 
import sys
import puzzle_solver as ps
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract' # windows only
tess_data_windows = 'C:\\Program\ Files\\Tesseract-OCR\\tessdata'

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
            
            #shadow removal from https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
            rgb_planes = cv.split(img)
            result_planes = []
            result_norm_planes = []
            for plane in rgb_planes:
                dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
                bg_img = cv.medianBlur(dilated_img, 21)
                diff_img = 255 - cv.absdiff(plane, bg_img)
                norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
                result_planes.append(diff_img)
                result_norm_planes.append(norm_img)

            result = cv.merge(result_planes)
            result_norm = cv.merge(result_norm_planes)
            return result_norm

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
    canny = cv.Canny(gray, 50, 150, None, 3)
    contours,hier = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("Didn't find any squares!")
        return
    
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
 
def tesseract(puzzle, bank) -> list:
    #resize
    puzzle = cv.resize(puzzle, (0,0), fx=3, fy=3)
    bank = cv.resize(bank, (0,0), fx=3, fy=3)
    
    #grayscale
    puzzle = cv.cvtColor(puzzle, cv.COLOR_BGR2GRAY)
    bank = cv.cvtColor(bank, cv.COLOR_BGR2GRAY)
    
    #apply otsu binarize
    cv.threshold(src=puzzle, thresh=100, maxval=255, dst=puzzle, type=cv.THRESH_OTSU)
    cv.threshold(src=bank, thresh=95, maxval=255, dst=bank, type=cv.THRESH_OTSU)
    
    #correct orientation
    puzzle = cv.rotate(puzzle, cv.ROTATE_90_CLOCKWISE)
    bank = cv.rotate(bank, cv.ROTATE_90_CLOCKWISE)
    
    display(puzzle)
    #display(bank)
    
    puzzle_config = r'--tessdata-dir "./protos_data/tessdata" -l eng  --oem 0 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ load_system_dawg=0 load_freq_dawg=0'
    bank_config = r'--tessdata-dir "./protos_data/tessdata_best" -l eng --oem 2 --psm 3'
    
    puzzle_detection = pytesseract.image_to_string(puzzle, config = puzzle_config)
    bank_detection = pytesseract.image_to_string(bank)
    
    #cleanup detection output
    parsed_puzzle = [i.strip() for i in puzzle_detection.split('\n') if len(i.strip()) > 0]
    #rotated_puzzle = list(zip(*parsed_puzzle[::-1]))
    parsed_bank = [i.strip() for i in bank_detection.split('\n') if len(i.strip()) > 2]
    
    #quick and messy bounding boxes
    d = pytesseract.image_to_data(puzzle, output_type=pytesseract.Output.DICT, config=puzzle_config)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv.rectangle(puzzle, (x, y), (x + w, y + h), (0, 255, 0), 2)
    display(puzzle)
    
    print('-------------------PUZZLE----------------------')
    #print('/n'.join(rotated_puzzle[0]))
    
    #clean_puzzle = rotated_puzzle[0]
    '''
    for j in range(len(clean_puzzle[0][0])):		
        for idx, item in enumerate(clean_puzzle):
            try:
                print(clean_puzzle[idx][0][j], end='')
            except:
                print('*', end='')
        print('')
    '''
            
    print(parsed_puzzle)
    print('-------------------WORD BANK----------------------')
    print(parsed_bank)
    
    return parsed_puzzle, parsed_bank
    
    
def debug(function, device):

    while True:
        img = capture_image(device, debug=True)
        output  = function(img, debug=True)
        cv.imshow('output', output)
        if cv.waitKey(1) == 27: 
            break;

    cv.destroyAllWindows()

def main():

    if int(str(pytesseract.get_tesseract_version())[0]) < 4:
        sys.exit('Tesseract 4.0.0 or greater required!') 

    cam = cv.VideoCapture(0)
    cam.set(3,1280) #height
    cam.set(4,720) #width

    img = capture_image(cam)

    puzzle, bank = segment(img)
    detected_puzzle, detected_bank = tesseract(puzzle, bank)
    solver = ps.PuzzleSolver(1,1, detected_puzzle, detected_bank)
    
if __name__ == '__main__':
    main()
