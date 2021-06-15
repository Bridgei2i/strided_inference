import cv2
import math
import numpy as np
import pandas as pd
import glob
import os
import shutil
from .tiler import tiler
from .detiler import detiler
import json


def strided_inference(image, filename, DT, tile_size_info = (1024, 600, 601), nms_th = 0.95):
    '''Strided Inference function takes the image and the detector 
    function as an input. For a given or specified size of tiles, 
    it performs strided inference over the overlapping tiles of the 
    image, detiles the results, perform NMS and gives back consolidated 
    results in form of a pd.DataFrame.

    NOTE:
    The detector function should be a python function that takes a 
    dictionary where key is image name(with file format) and value 
    is the image as numpy array. It returns detections and useful 
    information in form of a pd.DataFrame. The dataframe header 
    should be of the form:
    ['filename', 'label', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'].
    For more information, checkout the helper notebook provided.
    
    Parameters
    ----------
    image : np.array
        Image in form of numpy array.
    filename : str
        Image name to create uniquely named temporary folder, later deleted.
    DT : func
        A detection function that takes a dictionary where key is
        image name(with file format) and value is the image as numpy
        array. It returns back detections. Check NOTE above for more
        detail.
    tile_size_info : tuple
        A tuple containing information for the overlapping tiles
        needed to perform strided inferencing. Tuple values are
        (tile_size, offset, threshold).
    nms_th : float
        Threshold at which detiling will perform NMS to remove
        duplicate detections.

    Returns
    -------
    gt : pd.DataFrame
        DataFrame containing detection on the whole image.
    '''
    input_image = image.copy()
    
    tile_size, offset, threshold= tile_size_info
    
    obj = tiler()
    tile_img_dict, tile_ori_info = obj.tiling(input_image, filename, tile_size, offset, threshold)
    
    df = DT(tile_img_dict)
    
    if df.shape[0] < 1:
        gt = pd.DataFrame(columns=['filename', 'label', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'])


    if df.shape[0] > 1:
        ob = detiler()
        gt = ob.detiling(df, tile_ori_info, nms_th)
        
    return gt