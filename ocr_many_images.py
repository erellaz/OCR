"""
Performs Optical Character Recognition on a directory of images
Using Tesseract
For Windows:
-install Tesseract from UB Mannheim, add appropriate languages, adjust binary path below
-pip install pytesseract
"""

import cv2
import pytesseract
import numpy as np
import os

# set up the path to the binary without messing with environment variables:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# directory with the images to OCR
image_folder = r'D:\Dropbox\Jezierski\07_Ponchardier'
image_folder = r'D:\Dropbox\Jezierski\08_Commando_Ponchardier'

# erosion kernel
kernel = np.ones((2, 1), np.uint8)

# imagelist is the list with all image filenames
imagelist = [img for img in os.listdir(image_folder) if img.endswith(".JPG")]

# loop on all the images:
for image_name in imagelist:
    print("Performing OCR on:",image_name)
    img = cv2.imread(os.path.join(image_folder,image_name))
    #Alternatively: can be skipped if you have a Blackwhite image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)
    # preprocess those images
    img = cv2.erode(gray, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    # let tesseract perform text recognition
    ocr_string = pytesseract.image_to_string(img,lang='fra')#, config='--tessdata-dir .')
    #print("OUTPUT:", ocr_string)
    # write to a text file
    with open(os.path.join(image_folder,"OCR.txt"), "a") as text_file:
        text_file.write(ocr_string)

