import os
import argparse
import importlib
import io
import matplotlib.pyplot as plt
import numpy as np
from openslide import open_slide
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageChops, ImageDraw
import SimpleITK as sitk
from skimage.color import rgb2hed,hed2rgb,rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
import staintools
from staintools import stain_normalizer, LuminosityStandardizer
from staintools.preprocessing.input_validation import is_uint8_image
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#Allow Pillow to open very big images
Image.MAX_IMAGE_PIXELS = None

# Import HEMnet package
BASE_DIR = Path().resolve()
HEMNET_DIR = BASE_DIR.joinpath('HEMnet')
sys.path.append(str(HEMNET_DIR))

from slide import *
from utils import *
from normaliser import IterativeNormaliser

#############
# Functions #
#############

def restricted_float(x):
    #Restrict argument to float between 0 and 1 (inclusive)
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError('{0} not a floating point literal'.format(x))
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError('{0} not in range [0.0, 1.0]'.format(x))
    return x

def dab_tile_array(img, tile_size):
    """Creates array with mean DAB intensity value for each tile

    Parameters
    ----------
    img : Pillow image (RGB)
    tile_size : int

    Returns
    -------
    dab_tile_array : ndarray
    """
    dab_values = []
    tgen = tile_gen(img, tile_size)
    shape = next(tgen)
    for tile in tgen:
        tile = tile.convert('RGB')
        tile_hed = rgb2hed(tile)
        tile_dab = -tile_hed[:,:,2]
        dab_values.append(tile_dab.mean())
    return np.reshape(dab_values, shape)

def uncertain_mask(img, tile_size, cancer_thresh, non_cancer_thresh):
    """Create mask of uncertain tiles

    Parameters
    ----------
    img : Pillow image (RGB)
    tile_size : int
    cancer_thresh : float
        DAB intensity threshold between 0 and 1.0 (inclusive)
        Below threshold is cancer
    non_cancer_thresh : float
        DAB intensity threshold between 0 and 1.0 (inclusive)
        Above threshold is non-cancer

    Returns
    -------
    uncertain_mask : ndarray
    """
    dab_array = dab_tile_array(img, tile_size)
    binary_mask = ((cancer_thresh < dab_array) & (dab_array < non_cancer_thresh))
    return (np.invert(binary_mask)).astype(np.uint8)


def save_train_tiles(path, tile_gen, cancer_mask, tissue_mask, uncertain_mask, prefix = ''):
    """Save tiles for train dataset

    Parameters
    ----------
    path : Pathlib Path
    tile_gen : tile_gen
    cancer_mask : ndarray
    tissue_mask : ndarray
    uncertain_mask : ndarray
    prefix : str (optional)

    Returns
    -------
    None
    """
    os.makedirs(path.joinpath('cancer'), exist_ok = True)
    os.makedirs(path.joinpath('non-cancer'), exist_ok = True)
    os.makedirs(path.joinpath('uncertain'), exist_ok = True)
    x_tiles, y_tiles = next(tile_gen)
    verbose_print('Whole Image Size is {0} x {1}'.format(x_tiles, y_tiles))
    i = 0
    cancer = 0
    uncertain = 0
    non_cancer = 0
    for tile in tile_gen:
        img = tile.convert('RGB')
        ###
        img_norm = normaliser.transform_tile(img)
        ###
        # Name tile as horizontal position _ vertical position starting at (0,0)
        tile_name = prefix + str(np.floor_divide(i,x_tiles)) + '_' +  str(i%x_tiles)
        if uncertain_mask.ravel()[i] == 0:
            img_norm.save(path.joinpath('uncertain', tile_name + '.jpeg'), 'JPEG')
            uncertain += 1
        elif cancer_mask.ravel()[i] == 0:
            img_norm.save(path.joinpath('cancer', tile_name + '.jpeg'), 'JPEG')
            cancer += 1
        elif tissue_mask.ravel()[i] == 0:
            img_norm.save(path.joinpath('non-cancer', tile_name + '.jpeg'), 'JPEG')
            non_cancer += 1
        i += 1
    verbose_print('Cancer tiles: {0}, Non Cancer tiles: {1}, Uncertain tiles: {2}'.format(cancer, non_cancer, uncertain))
    verbose_print('Exported tiles for {0}'.format(prefix))

def fit_hsv(image_hsv,dab_binary,n):
    dab = np.logical_not(dab_binary)
    h = image_hsv[:,:,0][dab]
    s = len(h)
    while True:
        mu = h.mean()
        sig = h.std()
        h = h[np.logical_and(h>=max(0,mu-n*sig),h<=min(1,mu+n*sig))]
        if len(h) == s:
            break
        s = len(h)
    # print(len(image_hsv[dab][0]))
    # h = image_hsv[:,:,0]
    # s = image_hsv[:,:,1]
    # v = image_hsv[:,:,2]
    # mu = (h[dab].mean(),s[dab].mean(),v[dab].mean())
    # sig = (h[dab].std(),s[dab].std(),v[dab].std())
    # fig = plt.figure()
    # # ax1 = axes3d.Axes3D(fig)
    # # ax1.scatter3D(image_hsv[dab][:,0],image_hsv[dab][:,1],image_hsv[dab][:,2], cmap='Blues')
    # plt.scatter(image_hsv[dab][:,0],image_hsv[dab][:,2], cmap='Blues')
    # fig.savefig('test3.png',dpi=300)
    mu = h.mean()
    sig = h.std()
    # print(len(h[dab]))
    return mu,sig
def b_mask(img,dab_thresh):
    hed = rgb2hed(img)
    image_hsv = rgb2hsv(img)
    # print(image_hsv[:,:,0].max())
    # print(image_hsv[:,:,1].max())
    # print(image_hsv[:,:,2].max())
    dab_channel = -hed[:,:,2]
    dab_binary = dab_channel > dab_thresh   
    dab_binary = remove_small_holes(dab_binary, area_threshold = 64) 
    # mu,sig = fit_hsv(image_hsv, dab_binary,3)
    # map_0 = np.logical_and(image_hsv[:,:,0]>=max(0,mu[0]-n*sig[0]),image_hsv[:,:,0]<=min(1,mu[0]+n*sig[0]))
    # map_1 = np.logical_and(image_hsv[:,:,1]>=max(0,mu[1]-n*sig[1]),image_hsv[:,:,1]<=min(1,mu[1]+n*sig[1]))
    # map_2 = np.logical_and(image_hsv[:,:,2]>=max(0,mu[2]-n*sig[2]),image_hsv[:,:,2]<=min(1,mu[2]+n*sig[2]))
    # print(mu,sig)
    map_0 = np.logical_and(image_hsv[:,:,0]>=0,image_hsv[:,:,0]<=0.5)
    # map_0 = np.logical_and(map_0, map_1)
    # map_0 = np.logical_and(map_0, map_2)
    # map_0 = np.logical_not(map_0)
    dab_binary = np.logical_not(dab_binary)
    dab_binary = np.logical_and(dab_binary, map_0)
    dab_binary = np.logical_not(dab_binary)
    #Remove background staining
    dab_binary_filtered = remove_small_holes(dab_binary, area_threshold = 64)
    return dab_binary_filtered

def extract_dab_threshold(img):
    downsample = max(img.size)/1000
    img_small_size = tuple([np.int(np.round(dim/downsample)) for dim in img.size])
    img_small = img.resize(img_small_size, resample = Image.BICUBIC)
    hed_small = rgb2hed(img_small)
    dab_thresh = threshold_otsu_masked(hed_small)  
    return dab_thresh
    
def b_mask(img):
    #Determine Dab threshold with a smaller 1000x1000 image
    downsample = max(img.size)/1000
    # print(img.size)
    img_small_size = tuple([np.int(np.round(dim/downsample)) for dim in img.size])
    img_small = img.resize(img_small_size, resample = Image.BICUBIC)
    hed_small = rgb2hed(img_small)
    dab_thresh = threshold_otsu_masked(hed_small)
    # hed_thresh = threshold_otsu(-hed_small[:,:,0])
    # verbose_save_img(binary2gray(-hed_small[:,:,0]>hed_thresh),
    #                     OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'test.jpeg'), 'JPEG')
    #Extract Dab channel (stain)
    hed = rgb2hed(img)
    image_hsv = rgb2hsv(img)
    # print(image_hsv[:,:,0].max())
    # print(image_hsv[:,:,1].max())
    # print(image_hsv[:,:,2].max())
    dab_channel = -hed[:,:,2]
    dab_binary = dab_channel > dab_thresh   
    dab_binary = remove_small_holes(dab_binary, area_threshold = 64) 
    # mu,sig = fit_hsv(image_hsv, dab_binary,3)
    # map_0 = np.logical_and(image_hsv[:,:,0]>=max(0,mu[0]-n*sig[0]),image_hsv[:,:,0]<=min(1,mu[0]+n*sig[0]))
    # map_1 = np.logical_and(image_hsv[:,:,1]>=max(0,mu[1]-n*sig[1]),image_hsv[:,:,1]<=min(1,mu[1]+n*sig[1]))
    # map_2 = np.logical_and(image_hsv[:,:,2]>=max(0,mu[2]-n*sig[2]),image_hsv[:,:,2]<=min(1,mu[2]+n*sig[2]))
    # print(mu,sig)
    map_0 = np.logical_and(image_hsv[:,:,0]>=0,image_hsv[:,:,0]<=0.5)
    # map_0 = np.logical_and(map_0, map_1)
    # map_0 = np.logical_and(map_0, map_2)
    # map_0 = np.logical_not(map_0)
    dab_binary = np.logical_not(dab_binary)
    dab_binary = np.logical_and(dab_binary, map_0)
    dab_binary = np.logical_not(dab_binary)
    #Remove background staining
    dab_binary_filtered = remove_small_holes(dab_binary, area_threshold = 64)
    return dab_binary_filtered

def grabcut(img_filtered):
    img_cv = np.array(img_filtered)[:, :, ::-1]   #Convert RGB to BGR
    mask_initial = (np.array(img_filtered.convert('L')) != 255).astype(np.uint8)
    # Grabcut
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv.grabCut(img_cv, mask_initial, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    mask_final = np.where((mask_initial==2)|(mask_initial==0),0,1).astype('uint8')
    # Generate a rough 'filled in' mask of the tissue
    kernal_64 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (64,64))
    mask_closed = cv.morphologyEx(mask_final, cv.MORPH_CLOSE, kernal_64)
    mask_opened = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernal_64)
    # Use rough mask to remove small debris in grabcut mask
    mask_cleaned = cv.bitwise_and(mask_final, mask_final, mask = mask_opened)
    return np.logical_not(mask_cleaned.astype(np.bool))



BASE_PATH = "/dssg/home/acct-zhaochaoxian/zhaochaoxian-user1/ws_lmx/dataset/"
verbose_save_img = lambda img, path, img_type: img.save(path, img_type)
        
def process_mask(mask):
    #Three types mask have the same file name
    for x in mask[0]:
        cd = []
        cd.append(cv2.imread(os.path.join(BASE_PATH,"masks","CD3",x)))
        cd.append(cv2.imread(os.path.join(BASE_PATH,"masks","CD8",x)))
        cd.append(cv2.imread(os.path.join(BASE_PATH,"masks","CD20",x)))
        he_norm = cv2.imread(os.path.join(BASE_PATH,"patches","HE",x))
        cd_filtered = [filter_green(cd[i]) for i in range(len(cd))]
        he_filtered = filter_green(he_norm)
        cd_filtered = [filter_grays(cd_filtered[i], tolerance=2) for i in range(len(cd_filtered))]
        he_filtered = filter_grays(he_filtered, tolerance=15)
        # mask = [b_mask(cd_filtered[i].convert('RGB')) for i in range(len(cd_filtered))]
        dab_thre = [extract_dab_threshold(cd_filtered[i]) for i in range(len(cd_filtered))]
        mag_scale = get_amp_scale(he_slide,ALIGNMENT_MAG)
        downsample = max(he_filtered.size)/1000
        img_small_size = tuple([np.int(np.round(dim/downsample)) for dim in he_filtered.size])
        img_small = he_filtered.convert('RGB').resize(img_small_size, resample = Image.BICUBIC)
        hed_small = rgb2hed(img_small)
        he_hed = rgb2hed(he_filtered.convert('RGB'))
        h = -he_hed[:,:,0]
        threshold = threshold_otsu(-hed_small[:,:,0])
        he_h = h>threshold
        he_h = np.logical_or(he_h,grabcut(he_filtered.convert('RGB')))
        remove_small_holes(he_h, area_threshold = 64)
        # he_f = np.array(he_filtered.convert('RGB'))
        # he_h = np.logical_or(he_h,np.logical_and(np.logical_and(he_f[:,:,0]>200,he_f[:,:,1]>200),he_f[:,:,2]>200))
        # he_e = e>threshold_otsu(-hed_small[:,:,1])
        # verbose_save_img(binary2gray(np.array(he_filtered.convert('RGB'))==[255,255,255]),
        #                  OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'right.jpeg'), 'JPEG')
        downsample = max(he_filtered.size)/1000
        img_small_size = tuple([np.int(np.round(dim/downsample)) for dim in he_filtered.size])
        verbose_save_img(he_filtered.convert('RGB').resize(img_small_size, resample = Image.BICUBIC),
                            OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'he.jpeg'), 'JPEG')
        # verbose_save_img(binary2gray(he_h),
        #                  OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'he_mask.jpeg'), 'JPEG')
        # verbose_save_img(binary2gray(he_e),
        #                     OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'he_e_mask.jpeg'), 'JPEG')
        for i in range(len(cd_filtered)):
            downsample = max(cd_filtered[i].size)/1000
            img_small_size = tuple([np.int(np.round(dim/downsample)) for dim in cd_filtered[i].size])
            verbose_save_img(cd_filtered[i].convert('RGB').resize(img_small_size, resample = Image.BICUBIC),
                            OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + '{}.jpeg'.format(slide_list[i])), 'JPEG')
            verbose_save_img(binary2gray(np.logical_not(np.logical_and(np.logical_not(he_h),
                                                    np.logical_or(np.logical_or(np.logical_not(mask[0]),
                                                                                np.logical_not(mask[1])),
                                                                    np.logical_not(mask[2]))))),
                            OUTPUT_PATH.joinpath(PREFIX + str(ALIGNMENT_MAG) + 'he_all_mask.jpeg'), 'JPEG')
        
        mask = binary2gray(np.logical_not(np.logical_and(np.logical_not(he_h),
                                                    np.logical_or(np.logical_or(np.logical_not(mask[0]),
                                                                                np.logical_not(mask[1])),
                                                                    np.logical_not(mask[2])))))
        os.makedirs(os.path.join(BASE_PATH,"processed_mask"),exist_ok = True)
        cv2.imwrite(os.path.join(BASE_PATH,"processed_mask"),mask)
        
        
if __name__ == "__main__":
    MASK_PATH = os.path.join(BASE_PATH,"masks")
    CD3 = sorted(os.listdir(os.path.join(MASK_PATH+"CD3")))
    CD8 = sorted(os.listdir(os.path.join(MASK_PATH+"CD8")))
    CD20 = sorted(os.listdir(os.path.join(MASK_PATH+"CD20")))
    process([CD3,CD8,CD20])