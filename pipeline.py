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
import image2world as i2w
import image2world_test as i2wt
import argparse
import jamspell

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract' # windows only
#tess_data_windows = 'C:\\Program\ Files\\Tesseract-OCR\\tessdata'

word_swap = {
    "E": "F",
    "F": "E",
    "M": "H",
    "H": "M",
    "C": "G",
    "G": "C",
    "O": "Q",
    "Q": "O",
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


def segment(img, debug=False):
    img = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = np.ones(img.shape[:2], dtype='uint8') * 255 

    #ret,thresh = cv.threshold(gray,127,255,0)
    erosion_kernel = np.ones((3,3), np.uint8)
    erode = cv.erode(gray, erosion_kernel)
    canny = cv.Canny(erode, 50, 150, None, 3)
    #display(canny)
    contours,hier = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("Didn't find any contours!")
        return
    
    for i, cnt in enumerate(contours):
        if cv.contourArea(cnt)>50000 and cv.contourArea(cnt) <200000:  # only grab large contours
            print(cv.contourArea(cnt))
            hull = cv.convexHull(cnt) # find the convex hull of contour
            hull = cv.approxPolyDP(hull,0.1*cv.arcLength(hull,True),True)
            if len(hull)==4:
                cv.drawContours(img,[hull],0,(0,255,0),2)
                x,y,w,h = cv.boundingRect(cnt) # get minimal bounding for contour
                cv.drawContours(mask, [cnt], -1, 0, -1)
                
            #print(len(hull))
    if debug:
        display(img, 'contour')

    puzzle = img[y:y+h,x:x+w].copy()
    bank = cv.bitwise_and(img, img, mask=mask).copy()

    return puzzle, bank, x, y
 
def tesseract(puzzle, bank, x_offset, y_offset, debug=False, img = None) -> list:

    original_puzzle = puzzle
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

    #deskew_angle = math.ceil(deskew_angle) if deskew_angle > 0 else math.floor(deskew_angle)

    print('deskew', deskew_angle)
    M = cv.getRotationMatrix2D(center, deskew_angle, 1.0)
    puzzle = cv.warpAffine(puzzle, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    #if debug:
    #    display(puzzle, 'Preprocessed Puzzle')
    #    display(bank, 'Preprocessed Word Bank') 
    
    puzzle_config = r'--tessdata-dir "./protos_data/tessdata" -l eng  --oem 0 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ load_system_dawg=0 load_freq_dawg=0 textord_heavy_nr=1'
    bank_config = r' --oem 3 --psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ textord_heavy_nr=1 '
    
    puzzle_detection = pytesseract.image_to_string(puzzle, config = puzzle_config)
    bank_detection = pytesseract.image_to_string(bank, config=bank_config)
    
    #cleanup detection output
    parsed_puzzle = [i.strip() for i in puzzle_detection.split('\n') if len(i.strip()) > 0]
    #rotated_puzzle = list(zip(*parsed_puzzle[::-1]))
    parsed_bank = []
    overgrown_wordbank = []
    for i in bank_detection.split('\n'):
        if len(i.strip()) > 2:
            overgrown_wordbank.extend(i.strip().split())
            
    for word in overgrown_wordbank:
        if len(word) > 2:
            parsed_bank.append(word)
    parsed_bank.append('POOP')
    # quick and messy bounding boxes
    # d = pytesseract.image_to_data(puzzle, output_type=pytesseract.Output.DICT, config=puzzle_config)
    boxes = pytesseract.image_to_boxes(puzzle, config = puzzle_config)
    #n_boxes = len(d['level'])
    
    print('-------------------PUZZLE----------------------')
            
    for i in range(len(parsed_puzzle)):
        if ' ' in parsed_puzzle[i]:
            parsed_puzzle[i] = parsed_puzzle[i].replace(' ', 'I').ljust(15, 'I')[0:15]
    
    print(parsed_puzzle)
    print('-------------------WORD BANK----------------------')
    print(parsed_bank)
    

        
    
    letters_per_row = len(parsed_puzzle[0])
    character_coords = []
    i = 0

    for b in boxes.splitlines():
        if i % 15 == 0:
            character_coords.append([])
        # tuple with top left and bottom right coords
        b = b.split(' ')
        bounds = (int(b[1]),h - int(b[2]),int(b[3]),h - int(b[4]))
        character_coords[-1].append( [(int(bounds[2]) + int(bounds[0]))//2, (int(bounds[1]) + int(bounds[3]))//2, b[0]])
        cv.rectangle(puzzle, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0,255,0), 2)
        i += 1
        

    upright_puzzle = [ [[0,0,0,''] for _ in range(w)] for _ in range(h)]
    
    print('-------------------ROTATING----------------------')
    
    #recalculate center for copy of image
    center = (w // 2, h // 2)
    #undo rotation and deskew rotation done previously
    M = cv.getRotationMatrix2D(center, 90 - deskew_angle, 1.0)
    #rotate centers of characters
    #puzzle = cv.warpAffine(puzzle, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    length = 0
    print(x_offset, y_offset)
    print(character_coords)
    
    for x in range(len(character_coords[0])):
        for y in range(len(character_coords)):
            #print(x,y,character_coords[x][y])
            orig = (character_coords[x][y][0], character_coords[x][y][1])
            new = rotatepoint(center, orig, -90-deskew_angle)
            character_coords[x][y][0] = new[0]//3 + x_offset 
            character_coords[x][y][1] = new[1]//3 + y_offset
            if debug:
                cv.circle(img, (character_coords[x][y][0], character_coords[x][y][1]), 1, (0,0,0), 2)
                
    if debug:
        display(img, 'FINAL OUTPOUT')   

    #if debug:
    #    display(puzzle, 'Bounding Box Output') 
    
   
              
    # list(zip(*character_coords))[::-1] rotate -90
    return parsed_puzzle, parsed_bank, character_coords


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
    parser.add_argument('-e', '--everything', help='Runs full end to end pipeline, expecting connection on UART', action='store_true')
    args = parser.parse_args()

    return args


def permutative_solve(detected_bank, detected_puzzle=None):

    #detected_bank = [word.lower() for word in detected_bank]

    if detected_puzzle == None:
        detected_puzzle = []
        with open('test_searches/word_search.txt', 'r') as f:
            line = f.readline()
            while line:
                detected_puzzle.append(line[:-1].upper())
                detected_puzzle[-1] = detected_puzzle[-1].split(' ')
                line = f.readline()
    else:
        size = len(detected_puzzle)
        temp_puzzle = [['|' for _ in range(size)] for _ in range(size)]
        for i in range(len(detected_puzzle)):
            for j in range(len(detected_puzzle[i])):
                temp_puzzle[i][j] = detected_puzzle[i][j]
        print(temp_puzzle)

    solver = ps.PuzzleSolver(
        len(detected_puzzle[0]), len(detected_puzzle), detected_puzzle, detected_bank
    )

    print('-------------------SOLVING PUZZLE----------------------')
    incorrect_words, found = solver.solve()

    if len(incorrect_words) == 0:
        print('\nALL WORDS FOUND!')
        return found

    print('-----------------RETRYING WITH SWAPPED CHARACTERS----------------')
    potential_words = []

    for word in incorrect_words:
        permutations = letter_swap(word)
        permutations.append(word)
        potential_words.append(permutations)

    print(potential_words)

    for word in solver.potential_words_solve(incorrect_words, potential_words):
        found.append(word)

    if len(incorrect_words) == 0:
        print('\nALL WORDS FOUND!')
        return found

    print('-----------------RETRYING WITH NEW BANK----------------')
    corrector = jamspell.TSpellCorrector()
    corrector.LoadLangModel('protos_data/en.bin')

    potential_words = []

    # get word candidates that may be the correct target word, put them in potential_words
    lower_case_attempts = []
    for word in incorrect_words:
        lower_case_attempts.append(word.lower())
        
    for i in range(len(lower_case_attempts)):
        candidates = list(corrector.GetCandidates(lower_case_attempts, i))
        candidates = [word.upper() for word in candidates]
        candidates.append(incorrect_words[i])
        potential_words.append(candidates)

    if len(potential_words) == 0:
        print(
            '\nCould not find alternate spellings for the following words: ',
            incorrect_words,
        )
        return found
    else:
        print('Incorrect words', incorrect_words)

    for word in solver.potential_words_solve(incorrect_words, potential_words):
        found.append(word)

    # Found every word! Done
    if len(incorrect_words) == 0:
        print('\nALL WORDS FOUND!')
        return found

    print('SOME WORDS NOT FOUND:', incorrect_words)
    return found

def rotatepoint(center, point, ang):
    trans_point = (point[0] - center[0], point[1] - center[1])
    
    x = math.cos((ang*math.pi)/180)*trans_point[0]-math.sin((ang*math.pi)/180)*trans_point[1]
    y = math.sin((ang*math.pi)/180)*trans_point[0]+math.cos((ang*math.pi)/180)*trans_point[1]
    
    return (int(round(x,0))+center[0],int(round(y,0))+center[1])
    

def main():
    args = parse_args()

    if int(str(pytesseract.get_tesseract_version())[0]) < 4:
        sys.exit('Tesseract 4.0.0 or greater required!')
    
    if args.everything:
        jetson_UART = "/dev/ttyTHS1"
        drawer = drw.Drawer(jetson_UART)
 
    cam = cv.VideoCapture(1, cv.CAP_V4L2)
    cam.set(3, 1280)  # height
    cam.set(4, 720)  # width

    
    
    xyz = capture_image(cam)
    xyz_params = chessboard_calibrate('calibration_dummy', 6, 8, debug=False)
    ret, mtx, dist, rvecs, tvecs = xyz_params
    h, w = xyz.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    xyz = cv.undistort(xyz, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    xyz = xyz[y:y+h, x:x+w]
    xy_check = i2wt.uv_to_xy(xyz, xyz_params,[],True)
    display(xy_check[0])
    print(xy_check[1])
    
    while True:
        try:
            if args.image:
                img = cv.imread(args.image)
            else:
                img = capture_image(cam)
            
            # Camera Calibration
            # param order ret, mtx, dist, rvecs, tvecs
            if not args.image:
                params = chessboard_calibrate('calibration', 6, 8, debug=False)
                ret, mtx, dist, rvecs, tvecs = params
                h, w = img.shape[:2]
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
                img = cv.undistort(img, mtx, dist, None, newcameramtx)
                x, y, w, h = roi
                img = img[y:y+h, x:x+w]
           
                #display(img, 'Calibration Output')
                
                
            img = remove_shadow(img)
            puzzle, bank, x_offset, y_offset = segment(img, True)
            
            detected_puzzle, detected_bank, char_coords = tesseract(puzzle, bank, x_offset, y_offset, debug=True, img=img)
            solved_word_points = permutative_solve(detected_bank, detected_puzzle)
            print(solved_word_points)
            solved_uv_points = i2wt.wordsearch_to_uv(char_coords, solved_word_points)
            
            print(solved_uv_points)#char_coords[solved_uv_points[0][0][0]][solved_uv_points[0][0][1]][2])
            
            #solved_uv_points = [[[[468 ], [222]],[[470],[642]]], [[[764],[446]], [[1064],[220]]]]
            to_MSP_points = i2wt.uv_to_xy(xyz, xyz_params, solved_uv_points, False)
            display(to_MSP_points[0])
            if args.everything:
                scaling_factor_x = 0.22
                scaling_factor_y = 0.22
                start_offset_x = 3.75
                start_offset_y = 6.6
                drawer.read(1)
                for point_pair in to_MSP_points[1]:
                    x1 = int(round((point_pair[0][0]+start_offset_x)/scaling_factor_x))
                    y1 = int(round((point_pair[0][1]+start_offset_y)/scaling_factor_y))
                    x2 = int(round((point_pair[1][0]+start_offset_x)/scaling_factor_x))
                    y2 = int(round((point_pair[1][1]+start_offset_y)/scaling_factor_y))
                    to_draw = [(x1, y1), (x2, y2)]
                    drawer.draw(to_draw)
                    drawer.read(1)
                drawer.send(255)
                cv.destroyAllWindows()
        except Exception as e:
            print(e)
        except (KeyboardInterrupt):
            print('See ya later!')
            if args.everything:
                drawer.cleanup()
            break
if __name__ == '__main__':
    main()
