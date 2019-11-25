import cv2 as cv
import pytesseract
from pytesseract import Output

# filename = 'words_binarize.png'
filename = "sb.png"
img = cv.imread(filename)

custom_oem_psm_config = r"--oem 1 --tessdata-dir ./"
print(pytesseract.image_to_string(img, config=custom_oem_psm_config))

d = pytesseract.image_to_data(
    img, output_type=Output.DICT, config=custom_oem_psm_config
)
n_boxes = len(d["level"])
for i in range(n_boxes):
    (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv.imshow("img", img)
cv.imwrite("out.png", img)
# cv.waitKey(0)
