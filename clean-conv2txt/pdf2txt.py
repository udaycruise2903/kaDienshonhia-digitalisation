"""
Script for converting pdfs to text files using tesseract.

Usage:
    pdf2txt.py --input-folder-path=<if> [--output-folder-path=<of>]

Options:
    --input-folder-path=<if>            Provide folder path of pdfs to be converted
    --output-folder-path=<op>           Specify output path for the text files
"""
from docopt import docopt
from pdf2image import convert_from_path
import tempfile
from tesserocr import PyTessBaseAPI, PSM
from tqdm import tqdm
import os
import cv2
import numpy
from PIL import Image

## TODO: Add language support. Right now, its defaulted to English and Assamese at the moment.
custom_config = r"--oem 1 -l Latin"

def converter(ifolder: str, ofolder: str):
    """
    convert pdf to text file using tesseract
    :params ifolder: provide input folder path
    :params ofolder: provide output folder path
    :params langs: provide the languages that you would like to recognise
    """
    for filename in os.listdir(ifolder):
        with tempfile.TemporaryDirectory() as path:
            images = convert_from_path(ifolder + filename, thread_count=8, output_folder=path)
            ofile = filename.split(".")[0] + ".txt"
            with PyTessBaseAPI(path='/home/script/tesseract/tessdata/best/.', lang='eng+spa+fra') as api:
                with open(ofolder + ofile, "w+") as f:
                    for img in tqdm(images):
                        # img = numpy.array(img)
                        # gray = get_grayscale(img)
                        # thresh = thresholding(gray)
                        # thresh = Image.fromarray(thresh)
                        api.SetImage(img)
                        f.write(api.GetUTF8Text())

## get gray scale
#def get_grayscale(image):
#    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## noise removal
#def remove_noise(image):
#    return cv2.medianBlur(image,5)

##thresholding
#def thresholding(image):
#    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

##dilation
#def dilate(image):
#    kernel = np.ones((5,5),np.uint8)
#    return cv2.dilate(image, kernel, iterations = 1)

##erosion
#def erode(image):
#    kernel = np.ones((5,5),np.uint8)
#    return cv2.erode(image, kernel, iterations = 1)

##opening - erosion followed by dilation
#def opening(image):
#    kernel = np.ones((5,5),np.uint8)
#    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

##canny edge detection
#def canny(image):
#    return cv2.Canny(image, 100, 200)

##skew correction
#def deskew(image):
#    coords = np.column_stack(np.where(image > 0))
#    angle = cv2.minAreaRect(coords)[-1]
#    if angle < -45:
#        angle = -(90 + angle)
#    else:
#        angle = -angle
#    (h, w) = image.shape[:2]
#    center = (w // 2, h // 2)
#    M = cv2.getRotationMatrix2D(center, angle, 1.0)
#    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#    return rotated

##template matching
#def match_template(image, template):
#    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


if __name__ == "__main__":
    args = docopt(__doc__)

    ifolder = args["--input-folder-path"]

    ofolder = args["--output-folder-path"]

    if not ofolder:
        ofolder = os.path.dirname(os.path.realpath(__file__))

    converter(ifolder, ofolder)
