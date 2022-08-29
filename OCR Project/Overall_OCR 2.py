#for pytesseract
#conda install -c conda-forge pytesseract
#https://medium.com/analytics-vidhya/performing-optical-character-recognition-with-python-and-pytesseract-using-anaconda-4bfe1ee6a75f
#for CV2
#conda install -c conda-forge opencv
#for Matlab
#https://stackoverflow.com/questions/46141631/running-matlab-using-python-gives-no-module-named-matlab-engine-error
import cv2 
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import matplotlib.pyplot as plt
import numpy as np
#import jiwer
import matlab.engine
import sys
import scipy.io
import Levenshtein
from scipy.ndimage import interpolation as inter
import math

#https://nanonets.com/blog/ocr-with-tesseract/
    
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

def deskew(img):
    img=np.asarray(img)
    bin_img=cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    #plot
    hist1, score1 = find_score(bin_img, 0)
    hist2, score2 = find_score(bin_img, angle)
    histo=[hist1,hist2]
    plt.rcdefaults()
    figure, ax = plt.subplots(nrows=1,ncols=2 )
    for ind,title in enumerate(histo):
        y_pos = np.arange(len(histo[ind]))
        ax.ravel()[ind].barh(y_pos,histo[ind])
        ax.ravel()[ind].set_xlabel('no. of black pixels')
        ax.ravel()[ind].set_ylabel('Image row number')
    plt.tight_layout()
    plt.show()
    
    print('Best angle: {}'.format(best_angle))
    # correct skew
    data = inter.rotate(img, best_angle, reshape=False, order=0)
    hist_white = np.sum(data==0, axis=1)
    print(data)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(hist_white))
    ax.barh(y_pos,hist_white)
    ax.set_xlabel('no. of white pixels')
    ax.set_ylabel('Image row number')
    plt.show()
    (h,w)=img.shape
    
    mw,mh=rotatedRectWithMaxArea(w, h, abs(math.radians(best_angle)))
    dw=math.ceil((w-mw)/2)
    dh=math.ceil((h-mh)/2)
    data=np.ascontiguousarray((data))
    print(w,h)
    print(mw,mh)
    print(dw,dh)
    #data=matlab.double(data[dh:h-dh,dw:w-dw].tolist())
    data=np.asarray(data[dh:h-dh,dw:w-dw])
    #.tolist()
    return data


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def rotatedRectWithMaxArea(w, h, angle):
  """
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
def OCReval(Actual,prediction):
    #Step 1:Preprocessing
    Actual=Actual.split("\n")
    prediction=prediction.split("\n")

    for idx, val in enumerate(prediction):
        prediction[idx]=val.split(" ")
    
    for idx, val in enumerate(Actual):
        Actual[idx]=val.split(" ")

    
    #Step 2:Evaluating Accuracy
    results=Actual.copy();
    for i, line in enumerate(Actual):
        for j, word in enumerate(line):
            try:
                Actual_word=list(word)
                OCR_word=list(prediction[i][j])
            except:
                results[i][j]=np.array([False] * len(Actual_word))
                continue
            if len(Actual_word)==len(OCR_word):
                try:
                    match=np.array(np.array(OCR_word)==np.array(Actual_word))
                    results[i][j]=match
                except:
                    results[i][j]=np.array([False] * len(Actual_word))
            else:
                match= [True if i in OCR_word else False for i in Actual_word] 
                results[i][j]=np.array(match)
    print(results)
    #Step 3:Generating Accuracy Resuts
    Total=0
    Score=0
    for i, line_result in enumerate(results):
        for j, word_result in enumerate(line_result):
            Total=Total+len(word_result)
            if False in word_result:
                Score=Score+sum(word_result)
            else:
                Score=Score+len(word_result)
    
    
    return (Score*100/Total)

#https://www.delftstack.com/howto/matplotlib/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/

def display_multiple_img(images, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title], cmap='gray')
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()

def tesseractOCR(img,sample_no):
    results=[0,0]
    img=np.asarray(img)
    if sample_no==1:
        Actual="""Parking: You may park anywhere on the campus where there are no signs prohibiting par-
king. Keep in mind the carpool hours and park accordingly so you do not get bloked in the
afternoon

Under School Age Children:While we love the younger children, it can be disruptive and
inappropriate to have them on campus during school hours. There may be special times
that they may be invited or can accompany a parent volunteer, but otherwise we ask that
you adhere to our policy for the benefit of the students and staff."""
    if sample_no==2:
        Actual="""A Sonnet for Lena
O dear Lena, your beauty is so vast
It is hard sometimes to describe it fast.
I thought the entire world I would impress
If only your portrait I could compress.
Alas! First when I tried to use VQ
I found that your cheeks belong to only you.
Your silky hair contains a thousand lines
Hard to match with sums of discrete cosines.
And for your lips, sensual and tactual
Thirteen Crays found not the proper fractal.
And while these setbacks are all quite severe
I might have fixed them with hacks here or there
But when filters took sparkle from your eyes
I said, "Heck with it.  I'll just digitize."

Thomas Colthurst"""
        
    custom_config = r'--oem 3 --psm 6'
    OCR_output=pytesseract.image_to_string(img, config=custom_config)
    print(OCR_output)
##    OCR_output="""Ypark anywhere on the campus where tel No sigt
##. . fs nd the carpool ho and park accordingly so you do not |
##
##Unters . Age Children. While we love ine younger children, 1 cal
##inappropriate to have them on campus during school hours. There” nay t
##hat they may be jpyited or can accompany a parent volunteer, buto v
##fou adhere to ou olicy for the benefit of the students and staff
##
##"""


    results[0]=OCReval(Actual,OCR_output)
    results[1]=Levenshtein.distance(Actual, OCR_output)
    #print("Accuracy:", '%.2f'%(results[0]),"%" )
    print("Levenshtein:", '%.2f'%(results[1]))
    return results

img = get_grayscale(cv2.imread('resource/sample01.png'))
imgf = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #imgf contains Binary image
img_deskew=deskew(img)

images = {'Original': img,'-1 $^\circ$': img_deskew}
display_multiple_img(images, 1, 2)
print(tesseractOCR(img_deskew,1))
