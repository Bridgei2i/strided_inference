import cv2
import math
import numpy as np
import pandas as pd
import glob
import os
import sys
import time

class tile_for_training():    
    def __init__(self):
        self.tile_size = 1024
        self.offset = 600
        self.threshold = 601
        self.out_dir = 'Tiles_out'
        self.inp_dir = None
    
    def __tiler(self, img_name, img_data):
        img = cv2.imread(self.inp_dir+'/'+img_name)
        img_shape = img.shape

        filenames, heights, widths, labels, xmins, xmaxs, ymins, ymaxs = ([] for p in range(8))

        for i in range(int(math.ceil(img_shape[0]/self.offset))):
            for j in range(int(math.ceil(img_shape[1]/self.offset))):
                x1 = self.offset*i
                x2 = min(x1+self.tile_size, img_shape[0])

                y1 = self.offset*j
                y2 = min(y1+self.tile_size, img_shape[1])

                if (x2-x1)<self.threshold:
                    x1 = img_shape[0] - self.tile_size

                if (y2-y1)<self.threshold:
                    y1 = img_shape[1] - self.tile_size

                cropped_img = img[x1:x2, y1:y2]
                
                #Don't save the tile if it is more than 99 percent white coloured or empty
                if np.sum(cropped_img>=250)/(cropped_img.shape[0]*cropped_img.shape[1]*3)>0.99:
                    continue

                fname = f"{img_name[:-4]}###" + str(i) + "_" + str(j) + ".jpg"
                cv2.imwrite(self.out_dir+'/tiled_images/'+fname, cropped_img)


                for row in img_data.itertuples(index = True):     
                    label, x1_label, x2_label, y1_label, y2_label = (getattr(row, "label"), getattr(row, "xmin"), \
                                                                     getattr(row, "xmax"), getattr(row, "ymin"), getattr(row, "ymax"))
                    if (x1_label>=y1) and (x2_label<=y2) and (y1_label>=x1) and (y2_label<=x2):
                        filenames.append(fname)
                        heights.append(cropped_img.shape[0])
                        widths.append(cropped_img.shape[1])
                        labels.append(label)
                        xmins.append(int(x1_label - y1))
                        xmaxs.append(int(x2_label - y1))
                        ymins.append(int(y1_label - x1))
                        ymaxs.append(int(y2_label - x1))

        return pd.DataFrame({'filename':filenames, 'width':widths, 'height':heights, 'label':labels, 'xmin':xmins, 
                            'xmax':xmaxs, 'ymin':ymins, 'ymax':ymaxs})
        
    def tiler_with_annotations(self, inp_dir, annotationCSV_path, out_dir = None, tile_size_info = (1024, 600, 601)):
        '''This intakes directory containing training images, path to the
        annotation CSV and the output directory. It creates the tiles of the
        given size specification and creates a new CSV for these new tiled 
        images for further training.
        
        Parameters
        ----------
        inp_dir : str
            Path to the directory containing all the images to train on.
        annotationCSV_path : str
            Path to the CSV containing all the annotation inoformation for
            each image in training. The CSV should in following format:
            [filename, label, xmin, ymin, xmax, ymax, xdiff, ydiff]
        out_dir : str
            Path to a directory or name to store the final images of tiles
            and the new annotation for these tiles. If not provided, output
            will be stored in a folder called Tile_out.
        tile_size_info : tuple
            A tuple containing information for the overlapping tiles
            needed to perform strided inferencing. Tuple values are
            (tile_size, offset, threshold).
        '''
        start_time = time.time()
        
        if out_dir is not None:
            self.out_dir = out_dir

            
        self.inp_dir = inp_dir
        self.tile_size, self.offset, self.threshold = tile_size_info
        
        img_ori_info = pd.DataFrame({'filename':[], 'width':[], 'height':[],'label':[], 'xmin':[],\
                                     'xmax':[], 'ymin':[], 'ymax':[]})

        anno_data = pd.read_csv(annotationCSV_path)
        images_to_work_with = anno_data["filename"].unique()
        images_in_input_dir = os.listdir(f'{inp_dir}/')
        
        
        print('Doing image check for available annotations:\n')
        
        flag = 0
        temp = set()
        for i in images_to_work_with:
            if i not in images_in_input_dir:
                temp.add(i)
                flag = 1
                
        if flag==1:
            print('Alert, following images not found')
            for im in temp: print(im)
            del temp
            sys.exit("Please add the images for available annotations")
        
        else:
            del temp
            print('All good!\n')

            
        out_tile_dir = os.path.join(self.out_dir,'tiled_images')
        out_csv_dir = os.path.join(self.out_dir,'tiled_annotation')
        
        
        for directory in (out_tile_dir, out_csv_dir):
            if not os.path.exists(directory):
                os.makedirs(directory)
          
        
        print("Start of tiling for each image\n")
        
        c=0
        for img_name in images_to_work_with:
            image_data = anno_data[(anno_data["filename"]==img_name)].reset_index(drop=True)
            a = self.__tiler(img_name, image_data)
            img_ori_info = img_ori_info.append(a, ignore_index=True, sort=False)

            print(img_name, " done")
            
            c+=1
            if c%10==0:
                print(f'\n{c} images done\n\n')
                
        #End of tiling process
        
        
        for col in ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']:
            img_ori_info[col] = img_ori_info[col].astype('int')


        fin = os.path.basename(annotationCSV_path)[:-4]

        img_ori_info.to_csv(os.path.join(out_csv_dir, fin+'_annotation_data.csv'), index=False)

        print("\nDONE\n")
        end_time =time.time()
        print(f'Time taken to run the whole process {round(end_time - start_time, 2)} seconds.')