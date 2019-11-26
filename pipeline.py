#!/usr/bin/env python3

import math 
import sys
import os
import argparse
import ast
import cv2 as cv
import pytesseract
import numpy as np
import puzzle_solver as ps
import draw as drw
import argparse
import jamspell

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract' # windows only
#tess_data_windows = 'C:\\Program\ Files\\Tesseract-OCR\\tessdata'

word_swap = {
    "g": "q",
    "i": "l",
    "h": "n",
    "q": "g",
    "l": "i",
    "n": "h",
    "c": "o",
    "o": "c",
    }


def binary_num(bin_str, max_length):

    l = len(bin_str)
    num = list(bin_str)
    i = l - 1

    while i >= 0:
        if num[i] == "0":
            num[i] = "1"
            break
        else:
            num[i] = "0"
        i -= 1

    bin_str = "".join(num)

    if i < 0:
        bin_str = "1" + bin_str

    if len(bin_str) > max_length:

        bin_str = ""

        for i in range(max_length):
            bin_str += "0"

    return bin_str


def letter_swap(word):
    letter_locations = []
    permutations = []

    for char in enumerate(word):
        if char[1] in word_swap.keys():
            letter_locations.append(char)

    cardinality = len(letter_locations)

    bin_str = ""
    for i in range(cardinality):
        if i == cardinality - 1:
            bin_str += "1"
            break

        bin_str += "0"

    while "1" in bin_str:

        perm = word
        for tup in enumerate(letter_locations):
            if bin_str[tup[0]] == "1":
                perm = perm[: tup[1][0]] + word_swap[tup[1][1]] + perm[tup[1][0] + 1 :]

        permutations.append(perm)
        bin_str = binary_num(bin_str, cardinality)
    return permutations


def capture_image(device, mirror=False, debug=False):
    
    while True:
        ret_val, img = device.read()
        if mirror: 
            img = cv.flip(img, 1)

        if debug:
            return img
        cv.imshow('Camera View', img)

        if cv.waitKey(1) == 27: 
            cv.destroyAllWindows()
            return img
            

def display(img, title='img'):
    h, w = img.shape[:2]
    if h > 1500 or w > 2000:
        img =  cv.resize(img, (0,0), fx=.25, fy=.25)
    while True:
        cv.imshow(title, img)
        if cv.waitKey(1) == 27: 
            cv.destroyAllWindows()
            break;

def calibrate(img, debug=False):
    pass

def chessboard_calibrate(path, chessboard_rows, chessboard_cols, debug=False):

    calibration_cache = os.path.join(path, 'calibration.txt')

    if not os.path.exists(path):
        print("Path doesn't exist!")
        return

    if os.path.exists(calibration_cache) and False:# and os.path.getsize(calibration_cache) > 0:
        print('Using previously calculated parameters!')
        params = tuple([ast.literal_eval(i) for i in open(calibration_cache, 'r').read().split('@')])
        return params

    objp = np.zeros((chessboard_cols*chessboard_rows,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_rows,0:chessboard_cols].T.reshape(-1,2)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        ext = os.path.splitext(filename)[1]
        if ext != '.png' and ext != '.jpeg' and ext != '.JPG':
            continue
        filepath = os.path.join(path, filename)

        print('Processing', filepath)
        img = cv.imread(filepath)

        if debug:
            img = cv.resize(img, (0,0), fx=.5, fy=.5)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (chessboard_rows, chessboard_cols),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            if debug:
                img = cv.drawChessboardCorners(img, (chessboard_rows, chessboard_cols), corners2,ret)
                display(img)

    
    params = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    mean_error = 0
    _, mtx, dist, rvecs, tvecs = params
    for i, val in enumerate(objpoints):
        imgpoints2, _ = cv.projectPoints(val, rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( 'Total calibration error: {}'.format(mean_error/len(objpoints)) )

    str_params = '@'.join(map(str,params))
    f = open(calibration_cache, 'w')
    f.write(str_params)

    return params

def remove_shadow(img):
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
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = np.ones(img.shape[:2], dtype='uint8') * 255 

    #ret,thresh = cv.threshold(gray,127,255,0)
    canny = cv.Canny(gray, 50, 150, None, 3)
    contours,hier = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("Didn't find any contours!")
        return
    
    for i, cnt in enumerate(contours):
        if cv.contourArea(cnt)>50000:  # only grab large contours
            print('contour:', i)
            hull = cv.convexHull(cnt) # find the convex hull of contour
            hull = cv.approxPolyDP(hull,0.1*cv.arcLength(hull,True),True)
            if len(hull)==4:
                #cv.drawContours(img,[hull],0,(0,255,0),2)
                x,y,w,h = cv.boundingRect(cnt) # get minimal bounding for contour
                cv.drawContours(mask, [cnt], -1, 0, -1)
                break
            #print(len(hull))

    puzzle = img[y:y+h,x:x+w]
    bank = cv.bitwise_and(img, img, mask=mask)

    if debug:
        display(bank, 'Masked Image')
    
    return puzzle, bank
 
def tesseract(puzzle, bank, debug=False) -> list:
    #resize
    puzzle = cv.resize(puzzle, (0,0), fx=3, fy=3)
    bank = cv.resize(bank, (0,0), fx=3, fy=3)
    
    #sharpen bank
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    bank = cv.filter2D(bank, -1, kernel)
    
    #grayscale
    puzzle = cv.cvtColor(puzzle, cv.COLOR_BGR2GRAY)
    bank = cv.cvtColor(bank, cv.COLOR_BGR2GRAY)

    #apply otsu binarize
    cv.threshold(src=puzzle, thresh=100, maxval=255, dst=puzzle, type=cv.THRESH_OTSU)
    cv.threshold(src=bank, thresh=95, maxval=255, dst=bank, type=cv.THRESH_OTSU)
    
    #correct initial orientation
    puzzle = cv.rotate(puzzle, cv.ROTATE_90_CLOCKWISE)
    bank = cv.rotate(bank, cv.ROTATE_90_CLOCKWISE)
    
    #deskew
    puzzle_not = cv.bitwise_not(puzzle)

    coords = np.column_stack(np.where(puzzle_not > 0))
    deskew_angle = cv.minAreaRect(coords)[-1]
    if deskew_angle < -45:
        deskew_angle = -(90 + deskew_angle)
    else:
        deskew_angle = -deskew_angle

    h, w = puzzle_not.shape[:2]
    center = (w // 2, h // 2)

    deskew_angle = math.ceil(deskew_angle) if deskew_angle > 0 else math.floor(deskew_angle)

    print('deskew', deskew_angle)
    M = cv.getRotationMatrix2D(center, deskew_angle, 1.0)
    #display(puzzle, 'puzzle before')
    puzzle = cv.warpAffine(puzzle, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    #display(puzzle, 'puzzle after')
    if debug:
        display(puzzle, 'Preprocessed Puzzle')
        display(bank, 'Preprocessed Word Bank')
    
    puzzle_config = r'--tessdata-dir "./protos_data/tessdata" -l eng  --oem 0 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ load_system_dawg=0 load_freq_dawg=0'
    bank_config = r' --oem 3 --psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ textord_heavy_nr=1 '
    
    puzzle_detection = pytesseract.image_to_string(puzzle, config = puzzle_config)
    bank_detection = pytesseract.image_to_string(bank, config=bank_config)
    
    #cleanup detection output
    parsed_puzzle = [i.strip() for i in puzzle_detection.split('\n') if len(i.strip()) > 0]
    #rotated_puzzle = list(zip(*parsed_puzzle[::-1]))
    parsed_bank = []
    for i in bank_detection.split('\n'):
        if len(i.strip()) > 2:
            parsed_bank.extend(i.strip().split())
    
    #quick and messy bounding boxes
    d = pytesseract.image_to_data(puzzle, output_type=pytesseract.Output.DICT, config=puzzle_config)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #cv.rectangle(puzzle, (x, y), (x + w, y + h), (0, 255, 0), 2)
        internal_boxes = len(parsed_puzzle[0]) 
        per_box_width = w // internal_boxes
        #print('pbw', per_box_width)

        for _ in range(internal_boxes):
            cv.rectangle(puzzle, (x, y), (x + per_box_width, y + h), (0, 255, 0), 2)
            x += per_box_width

    if debug:
        display(puzzle, 'Bounding Box Output')
    
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
    
    return parsed_puzzle, parsed_bank, deskew_angle, n_boxes
    
    
def debug(function, device):

    while True:
        img = capture_image(device, debug=True)
        output  = function(img, debug=True)
        cv.imshow('output', output)
        if cv.waitKey(1) == 27: 
            break;

    cv.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='word puzzle image to be solved', type=str)
    args = parser.parse_args()

    return args.image


def permutative_solve(detected_bank, detected_puzzle=None):

    detected_bank = [word.lower() for word in detected_bank]

    if detected_puzzle == None:
        detected_puzzle = []
        with open('test_searches/word_search.txt', 'r') as f:
            line = f.readline()
            while line:
                detected_puzzle.append(line[:-1].lower())
                detected_puzzle[-1] = detected_puzzle[-1].split(' ')
                line = f.readline()

    solver = ps.PuzzleSolver(
        len(detected_puzzle[0]), len(detected_puzzle), detected_puzzle, detected_bank
    )

    print('-------------------SOLVING PUZZLE----------------------')
    incorrect_words = solver.solve()

    if len(incorrect_words) == 0:
        print('\nALL WORDS FOUND!')
        return

    print('-----------------RETRYING WITH SWAPPED CHARACTERS----------------')
    potential_words = []

    for word in incorrect_words:
        permutations = letter_swap(word)
        permutations.append(word)
        potential_words.append(permutations)

    print(potential_words)

    solver.potential_words_solve(incorrect_words, potential_words)

    if len(incorrect_words) == 0:
        print('\nALL WORDS FOUND!')
        return

    print('-----------------RETRYING WITH NEW BANK----------------')
    corrector = jamspell.TSpellCorrector()
    corrector.LoadLangModel('protos_data/en.bin')

    potential_words = []

    # get word candidates that may be the correct target word, put them in potential_words
    for i in range(len(incorrect_words)):
        candidates = list(corrector.GetCandidates(incorrect_words, i))
        candidates.append(incorrect_words[i])
        potential_words.append(candidates)

    if len(potential_words) == 0:
        print(
            '\nCould not find alternate spellings for the following words: ',
            incorrect_words,
        )
        return
    else:
        print('Incorrect words', incorrect_words)

    solver.potential_words_solve(incorrect_words, potential_words)

    # Found every word! Done
    if len(incorrect_words) == 0:
        print('\nALL WORDS FOUND!')
        return

    print('SOME WORDS NOT FOUND:', incorrect_words)


def main():
    args = parse_args()

    if int(str(pytesseract.get_tesseract_version())[0]) < 4:
        sys.exit('Tesseract 4.0.0 or greater required!')
   
    jetson_UART = "/dev/ttyTHS1"
    drawer = drw.Drawer(jetson_UART)
 
    cam = cv.VideoCapture(0)
    cam.set(3, 1280)  # height
    cam.set(4, 720)  # width

    if args:
        img = cv.imread(args)
    else:
        img = capture_image(cam)

    # Camera Calibration
    # param order ret, mtx, dist, rvecs, tvecs
    params = chessboard_calibrate('calibration', 6, 8, debug=False)
    ret, mtx, dist, rvecs, tvecs = params
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    img = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
   
    #display(img, 'Calibration Output')

    img = remove_shadow(img)
    puzzle, bank = segment(img)
    '''
    while(True):
        drawer.send('00001010'.encode())
        data = drawer.read()
        print(data)
        #drawer.send(data)
    '''
       
    detected_puzzle, detected_bank, _, _ = tesseract(puzzle, bank, debug=True)
    #permutative_solve(detected_bank)
    drawer.cleanup()

if __name__ == '__main__':
    main()
