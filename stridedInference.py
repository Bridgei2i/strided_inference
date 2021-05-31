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

def stridedInference(input_image, filename, DT, tile_size_info = (1024, 600, 601)):
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
    

    df = DT(out_dir, output_save_dir)
    

    if df.shape[0] < 1:
        placeholder = pd.DataFrame(columns=['filename', 'label', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'])
        if os.path.exists(f'{filename[:-4]}_datam'):
            shutil.rmtree(f'{filename[:-4]}_datam')
        return placeholder

    if df.shape[0] > 1:
        ob = detiler()
        gt = ob.detiling(df, original_info_path)

        shutil.rmtree(f'{filename[:-4]}_datam')

        return gt