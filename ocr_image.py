import cv2
import pytesseract
import numpy as np

image_name=r'D:\Dropbox\Jezierski\04_Livret_militaire\DSC06495.JPG'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#tessdatadir=r'C:\Program Files\Tesseract-OCR\tessdata'

img = cv2.imread(image_name)
#Alternatively: can be skipped if you have a Blackwhite image
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.bitwise_not(img_bin)

kernel = np.ones((2, 1), np.uint8)
img = cv2.erode(gray, kernel, iterations=1)
img = cv2.dilate(img, kernel, iterations=1)
ocr_string = pytesseract.image_to_string(img,lang='fra')#, config='--tessdata-dir .')
print("OUTPUT:", ocr_string)