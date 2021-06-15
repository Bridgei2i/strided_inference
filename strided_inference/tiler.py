import cv2
import math
import numpy as np
import pandas as pd
import glob
import os


class tiler():
    """This is an intermediate class that has the tiling method
    which is used to create small overlapping tiles of a given
    image with either default or user-defined tile specifications.

    Methods
    -------
    tiling(img, img_name, output_dir, tile_size = 1024, offset = 600, threshold=601)
        Method to call to perform tiling. Check out method's 
        docstring for more information.
    """
    def __init__(self, img, img_name, out_dir, tile_size = 1024, offset = 600, threshold=601):        
        
        just_file_info = pd.DataFrame({'filename':[], 'width':[], 'height':[], 'original_origin_y1':[], 'original_origin_x1':[]})
        
        b = self.tiling(img, img_name, out_dir, tile_size, offset, threshold)
        just_file_info = just_file_info.append(b, ignore_index=True, sort=False)
        
        for col in ['width', 'height', 'original_origin_y1', 'original_origin_x1']:
            just_file_info[col] = just_file_info[col].astype('int')
            
        just_file_info.drop_duplicates(inplace=True)
        just_file_info.to_csv(os.path.join(out_dir,'..','temp_files','tile_original_info.csv'), index=False)

        
    def tiling(self, img, img_name, output_dir, tile_size = 1024, offset = 600, threshold=601):
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
        output_dir : str
            Path to a directory or name to store the final images of tiles.
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
        '''
        img_shape = img.shape

        filenames, heights, widths, labels, xmins, xmaxs, ymins, ymaxs = ([] for p in range(8))

        fi, he, wi, ori_y1, ori_x1 = ([] for pi in range(5))


        for i in range(int(math.ceil(img_shape[0]/offset))):
            for j in range(int(math.ceil(img_shape[1]/offset))):
                x1 = offset*i
                x2 = min(x1+tile_size, img_shape[0])

                y1 = offset*j
                y2 = min(y1+tile_size, img_shape[1])

                if (x2-x1)<threshold:
                    x1 = img_shape[0] - tile_size

                if (y2-y1)<threshold:
                    y1 = img_shape[1] - tile_size

                cropped_img = img[x1:x2, y1:y2]
                
                ### Start Of: Code block that ignores the 99% only white area
                
                if np.sum(cropped_img>=250)/(cropped_img.shape[0]*cropped_img.shape[1]*3)>0.99:
                    continue
                
                ### End
                
                fname = f"{img_name[:-4]}###" + str(i) + "_" + str(j) + ".jpg"
                cv2.imwrite(os.path.join(output_dir,fname), cropped_img)

                fi.append(fname)
                he.append(cropped_img.shape[0])
                wi.append(cropped_img.shape[1])
                ori_y1.append(int(y1))
                ori_x1.append(int(x1))

        return pd.DataFrame({'filename':fi, 'width':wi, 'height':he,'original_origin_y1':ori_y1, 'original_origin_x1':ori_x1})