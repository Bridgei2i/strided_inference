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


def stridedInference(image, filename, DT, tile_size_info = (1024, 600, 601), nms_th = 0.95):
    '''Strided Inference function takes the image and the detector 
    function as an input. For a given or specified size of tiles, 
    it performs strided inference over the overlapping tiles of the 
    image, detiles the results, perform NMS and gives back consolidated 
    results in form of a pd.DataFrame.

    NOTE:
    The detector function should be a python function that can take 
    a folder containing images as an input and returns detections 
    and useful information in form of a pd.DataFrame. The 
    dataframe header should be of the form:
    ['filename', 'label', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'].
    For more information, checkout the helper notebook provided.
    
    Parameters
    ----------
    image : np.array
        Image in form of numpy array.
    filename : str
        Image name to create uniquely named temporary folder, later deleted.
    DT : func
        A detection function that takes folder path as input
        and returns back detections. Check NOTE above for more
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
    
    if not os.path.exists(f'{filename[:-4]}_datam/tiles_out'):
        os.makedirs(f'{filename[:-4]}_datam/tiles_out')
   
    if not os.path.exists(f'{filename[:-4]}_datam/temp_files'):
        os.makedirs(f'{filename[:-4]}_datam/temp_files')        
    
    out_dir = f"{filename[:-4]}_datam/tiles_out"
    detection_path = f"{filename[:-4]}_datam/temp_files/detected_boxes.csv"
    original_info_path = f"{filename[:-4]}_datam/temp_files/tile_original_info.csv"
    output_save_dir = out_dir+'/../temp_files/detected_boxes'
    
    tile_size, offset, threshold= tile_size_info
    
    tiler(input_image, filename, out_dir, tile_size, offset, threshold)
    
    df = DT(out_dir)
    
    if df.shape[0] < 1:
        placeholder = pd.DataFrame(columns=['filename', 'label', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'])
        if os.path.exists(f'{filename[:-4]}_datam'):
            shutil.rmtree(f'{filename[:-4]}_datam')
        return placeholder

    if df.shape[0] > 1:
        ob = detiler()
        gt = ob.detiling(df, original_info_path, nms_th)

        shutil.rmtree(f'{filename[:-4]}_datam')

        return gt