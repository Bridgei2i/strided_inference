# Strided Inference: for small object detection in high resolution images
###     

![let.png](images/let.png)

![Header.png](images/Header.png)

###  [Link to Bi2i article](https://bridgei2i.com/strided_inferencing.com) | [Our arXiv paper](https://arxiv.org/abs/_____) | [Link to medium article on use case](https://amitamola.medium/Strided_inference) 

## Requirements
```bash
$ pip install -r requirements.txt
```


## 1. Using strided inference with a pre-trained model
```
>>> from stridedInference.stridedInference import stridedInference
>>> img_name = 'img.jpg'
>>> image = cv2.imread(f'openCV_dnn/{img_name}')

>>> detections = stridedInference(image, img_name, detector, tile_size_info=(900, 700, 701), nms_th=0.95)
```

## 2. To generate tiles for training new model 
The key use case of this module is to make use of Strided Inferencing with a pre-trained model, yet for lot of cases
there might be a need to train a new model on tiled images. We provide a submodule that performs this
task and outputs tiled images and new tiled CSV(containing annotations) for your new model training.

```
>>> from tiler_with_annotations import tile_for_training
>>> ob = tile_for_training()
>>> ob.tiler_with_annotations(image_folder_path, annotation_csv_path,  output_directory_path, tile_size_information)
```


## 3. Example 
Below we can see results of using OpenCV’s DNN module’s ssd_mobilenet network object detection with and without strided inference:

### Figure  1: Stock image(5306 x 2985) of a substation with a big crowd
![Original Stock Image](images/sample_img.jpg)

### Figure  2: Figure 2: Result of running object detector with low confidence
![Original Stock Image](images/result_without_strided.jpg)

### Figure 3: Detections using strided inference using 900 pixel tiles and NMS of 0.95
![Original Stock Image](images/result_with_strided.jpg)

There are some example cases:

 <img src="https://github.com/waittim/waittim.github.io/raw/master/img/mask-examples.jpg" width = "600"  alt="examples" align=center />

Hint: If you want to convert the model to the ONNX format (Not necessary), please check [**20-PyTorch2ONNX.ipynb**](https://github.com/waittim/mask-detector/blob/master/modeling/20-PyTorch2ONNX.ipynb)

## Deployment

The deployment part works based on NCNN and WASM.

### 1. Pytorch to NCNN
At first, you need to compile the NCNN library. For more details, you can visit [Tutorial for compiling NCNN library
](https://waittim.github.io/2020/11/10/build-ncnn/) to find the tutorial.

When the compilation process of NCNN has been completed, you can start to use various tools in the **ncnn/build/tools** folder to help us convert the model. 

For example, you can copy the **yolo-fastest.cfg** and **best.weights** files of the darknet model to the **ncnn/build/tools/darknet**, and use this code to convert to the NCNN model.

```bash
./darknet2ncnn yolo-fastest.cfg best.weights yolo-fastest.param yolo-fastest.bin 1
```

For compacting the model size, you can move the **yolo-fastest.param** and **yolo-fastest.bin** to **ncnn/build/tools**, then run the ncnnoptimize program.

```bash
ncnnoptimize yolo-fastest.param yolo-fastest.bin yolo-fastest-opt.param yolo-fastest-opt.bin 65536 
```
### 2. NCNN to WASM

Now you have the **yolo-fastest-opt.param** and **yolo-fastest-opt.bin** as our final model. For making it works in WASM format, you need to re-compile the NCNN library with WASM. you can visit [Tutorial for compiling NCNN with WASM
](https://waittim.github.io/2020/11/15/build-ncnn-wasm/) to find the tutorial. 

Then you need to write a C++ program that calls the NCNN model as input the image data and return the model output. The [C++ code](https://github.com/waittim/facemask-detection/blob/master/yolo.cpp) I used has been uploaded to the [facemask-detection](https://github.com/waittim/facemask-detection) repository. 

Compile the C++ code by `emcmake cmake` and `emmake make`, you can get the **yolo.js**, **yolo.wasm**, **yolo.worker.js** and **yolo.data**. These files are the model in WASM format.

### 3. Build webpage 
After establishing the webpage, you can test it locally with the following steps in the [facemask-detection](https://github.com/waittim/facemask-detection) repository:

1. start a HTTP server `python3 -m http.server 8888`
2. launch google chrome browser, open chrome://flags and enable all experimental webassembly features
3. re-launch google chrome browser, open http://127.0.0.1:8888/test.html, and test it on one frame.
4. re-launch google chrome browser, open http://127.0.0.1:8888/index.html, and test it by webcam.

To publish the webpage, you can use Github Pages as a free server. For more details about it, please visit https://pages.github.com/.


## Acknowledgement

The modeling part is modified based on the code from [Ultralytics](https://github.com/ultralytics/yolov3). The model used is modified from the [Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) model shared by dog-qiuqiu. Thanks to [nihui](https://github.com/nihui), the author of NCNN, for her help in the NCNN and WASM approach.