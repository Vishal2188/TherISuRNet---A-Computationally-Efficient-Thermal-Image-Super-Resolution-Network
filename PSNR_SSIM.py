import numpy as np
import numpy
import cv2
import glob
import os
from scipy import misc
from skimage.measure import compare_ssim
from skimage.color import rgb2ycbcr,rgb2yuv

from skimage.measure import compare_psnr
from os import listdir
from os.path import isfile, join

def luminance(image):
    lum = rgb2ycbcr(image)[:,:,0]
    lum = lum[4:lum.shape[0]-4, 4:lum.shape[1]-4]
    return lum

def PSNR(gt, pred):
    return compare_psnr(gt, pred, data_range=255)
    
def SSIM(gt, pred):
    ssim = compare_ssim(gt, pred, data_range=255, gaussian_weights=True)
    return ssim

mypath='Path to HR Images'
mypath1='Path to SR Images' 
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)

x=0
y=0
avg_psnr = 0
avg_ssim = 0
individual_psnr = []
individual_ssim = []
for i in range(0, len(onlyfiles)):
  gt=mypath+str(i+1)+'.jpg'
  pred=mypath1+str(i+1)+'.jpg'
  print(gt)
  print(pred)
  # compare to gt
  psnr = PSNR(luminance(misc.imread(gt, mode='RGB').astype(np.uint8)), luminance(misc.imread(pred, mode='RGB').astype(np.uint8)))
  ssim = SSIM(luminance(misc.imread(gt, mode='RGB').astype(np.uint8)), luminance(misc.imread(pred, mode='RGB').astype(np.uint8)))
  print("PSNR Value:",psnr)
  print("SSIM Value:",ssim)
  individual_psnr.append(psnr)
  individual_ssim.append(ssim)
  avg_psnr += psnr
  avg_ssim += ssim
    
avg_psnr /= i+1#len(pred)
avg_ssim /= i+1#len(pred)

print("PSNR Average value:", avg_psnr)
print("SSIM Average value:", avg_ssim)