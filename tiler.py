import cv2
import math
import numpy as np
import pandas as pd
import glob
import os


class tiler():
    def __init__(self, img, img_name, out_dir, tile_size = 1024, offset = 600, threshold=601):        
        
        just_file_info = pd.DataFrame({'filename':[], 'width':[], 'height':[], 'original_origin_y1':[], 'original_origin_x1':[]})
        
        b = self.tiling(img, img_name, out_dir, tile_size, offset, threshold)
        just_file_info = just_file_info.append(b, ignore_index=True, sort=False)
        
        for col in ['width', 'height', 'original_origin_y1', 'original_origin_x1']:
            just_file_info[col] = just_file_info[col].astype('int')
            
        just_file_info.drop_duplicates(inplace=True)
        just_file_info.to_csv(f'{out_dir}/../temp_files/tile_original_info.csv', index=False)
        
        
    def tiling(self, img, img_name, output_dir, tile_size = 1024, offset = 600, threshold=601):
        img_shape = img.shape
        
        #if any(np.array(img_shape)<tile_size):
        #    num = tile_size-min(img.shape)
        #    img = np.pad(img, (0, num), constant_values=255)[:max(img_shape[0], tile_size),:max(img_shape[1], tile_size)]
        
        #print(img.shape)

        '''
        if '/' in img_path:
            img_name = img_path.split('/')[-1]

        elif '//' in img_path:
            img_name = img_path.split('\\')[-1]
        
        else:
            img_name = img_path
        '''
        #img_name = os.path.basename(img_path)

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

                if np.sum(cropped_img>=250)/(cropped_img.shape[0]*cropped_img.shape[1]*3)>0.99:
                    continue

                fname = f"{img_name[:-4]}###" + str(i) + "_" + str(j) + ".jpg"
                cv2.imwrite(output_dir+'/'+fname, cropped_img)

                fi.append(fname)
                he.append(cropped_img.shape[0])
                wi.append(cropped_img.shape[1])
                ori_y1.append(int(y1))
                ori_x1.append(int(x1))

        return pd.DataFrame({'filename':fi, 'width':wi, 'height':he,'original_origin_y1':ori_y1, 'original_origin_x1':ori_x1})