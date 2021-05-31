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

def stridedInference(input_image, filename, DT):
    if not os.path.exists(f'{filename[:-4]}_datam/tiles_out'):
        os.makedirs(f'{filename[:-4]}_datam/tiles_out')
   
    if not os.path.exists(f'{filename[:-4]}_datam/temp_files'):
        os.makedirs(f'{filename[:-4]}_datam/temp_files')        
    
    out_dir = f"{filename[:-4]}_datam/tiles_out"
    detection_path = f"{filename[:-4]}_datam/temp_files/detected_boxes.csv"
    original_info_path = f"{filename[:-4]}_datam/temp_files/tile_original_info.csv"
    output_save_dir = out_dir+'/../temp_files/detected_boxes'
    tiler(input_image, filename, out_dir)
    

    DT(out_dir, output_save_dir)
    
    #NOTE- It's being assumed the detection would give a JSON as output
    
    #START of: Conversion of JSON to CSV
    with open(output_save_dir+'.json', 'r') as f:
        dat = json.load(f)

    df = pd.DataFrame(columns = ['filename', 'class', 'boxes', 'confidence'])
    for i in dat:
        temp_df = pd.DataFrame()
        dsize = len(dat[i]['boxes'])
        temp_df['filename'] = [i]*dsize
        temp_df['boxes'] = dat[i]['boxes']
        temp_df['class'] = dat[i]['classes']
        temp_df['confidence'] = dat[i]['confidence']
        res = pd.concat([res, temp_df], axis = 0, sort=False)    
    #END

    if df.shape[0] < 1:
        placeholder = pd.DataFrame(columns=['filename', 'class', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'])
        if os.path.exists(f'{filename[:-4]}_datam'):
            shutil.rmtree(f'{filename[:-4]}_datam')
        return placeholder

    if df.shape[0] > 1:
        ob = detiler()
        gt1 = ob.detiling(detection_path, original_info_path)

        shutil.rmtree(f'{filename[:-4]}_datam')

        return gt