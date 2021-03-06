import cv2
import math
import numpy as np
import pandas as pd
import glob
import os
from joblib import Parallel, delayed
import itertools


class tiler():
    """This is an intermediate class that has the tiling method
    which is used to create small overlapping tiles of a given
    image with either default or user-defined tile specifications.
    This is multiprocessing implementation of tiler using joblib.
    In case it causes issue, please use the other code named as
    tiler_wo_joblib.

    Methods
    -------
    tiling(img, img_name, output_dir, tile_size = 1024, offset = 600, threshold=601)
        Method to call to perform tiling. Check out method's 
        docstring for more information.
    """
    def __init__(self):        
        pass
        
    def tiling(self, img, img_name, tile_size = 1024, offset = 600, threshold=601):
        '''This method creates small overlapping tiles in the specified 
        output directory.
        
        NOTE: For any portion of the image when converted to tile, if more than
        99% of the tile is nothing but white coloured, it will be ignored.
        To ignore, comment out the code between start of and end.
        
        Parameters
        ----------
        img : np.array
            Image in form of numpy array.
        img_name : str
            Image name to create uniquely named temporary folder, later deleted.
        tile_size : int
            Integer denoting size of the tile.
        offset : int
            Amount of overlapping between the tiles. Ignored if tile size is
            below the threshold and making tile go out of bounds.
        threshold : int
            This defines the minimum size a tile can have if reached out of bounds.
            In case the coordinates goes beyond the dimensions, it will cut of the
            tile of the threshold size till the end of the either axis to 
            return/create a minimum threshold sized tile.
            
        Returns
        -------
        tile_dict : dict
            A dictionary with key as image name(with file format) and value 
            is the image as numpy array.
        ret : pd.DataFrame
            DataFrame containing coordinate information of tiles created.
        '''
        def process_img(i, j):
            temp_dict = dict()
            x1 = offset*i
            x2 = min(x1+tile_size, img_shape1)

            y1 = offset*j
            y2 = min(y1+tile_size, img_shape2)

            if (x2-x1)<threshold:
                x1 = img_shape1 - tile_size

            if (y2-y1)<threshold:
                y1 = img_shape2 - tile_size

            cropped_img = img[x1:x2, y1:y2]
                
            ### Start Of: Code block that ignores the 99% only white area
            if np.sum(cropped_img>=250)/(cropped_img.shape[0]*cropped_img.shape[1]*3)>0.99:
                return None
            
            else:
                fname = f"{img_name[:-4]}###" + str(i) + "_" + str(j) + ".jpg"
#                 cv2.imwrite(output_dir+'/ims/'+fname, cropped_img)
                temp_dict[fname] = cropped_img

                return [temp_dict, fname, cropped_img.shape[0], cropped_img.shape[1], int(y1), int(x1)]
                
        img_shape1, img_shape2, _ = img.shape
        tile_dict = dict()
        
        all_i = range(int(math.ceil(img_shape1/offset)))
        all_j = range(int(math.ceil(img_shape2/offset)))
        vals = itertools.product(all_i, all_j)
                
        df_list = Parallel(n_jobs=-1)(delayed(process_img)(i, j) for i,j in vals)
        df_list = [x for x in df_list if x is not None]
        
        for i in df_list:
            tile_dict.update(i.pop(0))
                
        ret = pd.DataFrame(df_list, columns=['filename', 'width', 'height','original_origin_y1', 'original_origin_x1'])
        
        for col in ['width', 'height', 'original_origin_y1', 'original_origin_x1']:
            ret[col] = ret[col].astype('int')
            
        ret.drop_duplicates(inplace=True)
            
        return tile_dict, ret