import pandas as pd
import numpy as np
import cv2

class detiler():
    def __init__(self):
        pass
        
    def detiling(self, tiled_csv, original_info_path, nms_th=0.95):
        original_info = pd.read_csv(original_info_path)

        new_tiled = pd.DataFrame(columns=["filename", "label", "xmin", "xmax", "ymin", "ymax", "confidence"])
        col_schema = new_tiled.columns
        
        for row in tiled_csv['filename'].unique():    
            fn, cl, xmi, xma, ymi, yma, sco = [[] for x in range(7)]
            temp_df = tiled_csv[(tiled_csv['filename']==row)].reset_index(drop=True).copy(deep=True)            
            values = original_info[original_info['filename']==row].values[0]

            width = values[1]
            height = values[2]

            origin_y1 = values[3]
            origin_x1 = values[4]

            for val in temp_df.itertuples():
                x_cordi = np.array((int(getattr(val, 'xmin')*width), int(getattr(val, 'xmax')*width)))
                y_cordi = np.array((int(getattr(val, 'ymin')*height), int(getattr(val, 'ymax')*height)))
        
                x_cordi += origin_y1
                y_cordi += origin_x1

                first_vertex = (min(x_cordi), min(y_cordi))
                second_vertex = (max(x_cordi), max(y_cordi))
  

                img_name = getattr(val, 'filename').split("###")[0] + '.jpg'
                cl.append(getattr(val, 'label'))
                sco.append(getattr(val, 'confidence'))                
                fn.append(img_name)
                
                xmi.append(first_vertex[0])
                xma.append(second_vertex[0])
                ymi.append(first_vertex[1])
                yma.append(second_vertex[1])

            added_df = pd.DataFrame({'filename':fn, "label":cl, "confidence":sco, "xmin":xmi, "xmax":xma, "ymin":ymi, "ymax":yma})    
            new_tiled = new_tiled.append(added_df, ignore_index=True, sort=False)

        final_out = pd.DataFrame(columns=col_schema)
        
        # START OF: Performing Non Max Suppresion over the detections we have got
        for filename in new_tiled['filename'].unique():
            tempi = new_tiled[(new_tiled['filename']==filename)]
            tempi = tempi[col_schema]

            for sym in tempi['label'].unique():
                tempi2 = (tempi[(tempi['label']==sym)]).reset_index(drop=True)
                tempi2 = tempi2[col_schema]

                box_vals = [(vall[3], vall[4], vall[5], vall[6]) for vall in tempi2.itertuples()]
                chk_scores = [0.9 for a in range(len(box_vals))]

                try:
                    indexes = cv2.dnn.NMSBoxes(box_vals, chk_scores, score_threshold=0.5, nms_threshold=nms_th)
                except:
                    continue

                indexes  = [index[0] for index in indexes]        
                outTable = tempi2.iloc[indexes]
                final_out = final_out.append(outTable, ignore_index=True, sort=False)
        # END OF: NMS
        
        return final_out