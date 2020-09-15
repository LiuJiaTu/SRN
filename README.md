# Eelective Refinement Network for High Performance Face Detection
this is third-party train models useing mxnet,the official is [here](https://github.com/ChiCheng123/SRN/blob/master/README.md#selective-refinement-network-for-high-performance-face-detection)
## Data
1.Download annotations(face bounding boxes) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](http://dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0),provided by [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace).</br>
2.Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.</br>
3.Organise the dataset directory under SRN/ as follows:</br>
>data/widerface/</br>

>>train/</br>
>>>images/</br>
>>>label.txt</br>

>>val/</br>
>>>images/</br>
>>>label.txt</br>

>>test/</br>
>>>images/</br>
>>>label.txt</br>
## Install
1.Install MXNet with GPU support.</br>
2.Type `make` to build cxx tools.(it means input `make` in linux terminal)
## Train
1.Download ImageNet pretrained models and put them into `model/`(these models are not for detection testing/inferencing but training and parameters initialization).
    
　　ImageNet ResNet50 ([baidu cloud](https://pan.baidu.com/s/1nzQ6CzmdKFzg8bM8ChZFQg) and [dropbox](https://www.dropbox.com/s/8ypcra4nqvm32v6/imagenet-resnet-152.zip?dl=0)).</br>
　　ImageNet ResNet152 ([baidu cloud](https://pan.baidu.com/s/1nzQ6CzmdKFzg8bM8ChZFQg) and [dropbox](https://www.dropbox.com/s/8ypcra4nqvm32v6/imagenet-resnet-152.zip?dl=0)).　Provided by [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace).</br>
2.Start training with `CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py`.(more configuration in `rcnn/config.py`.)
## Test
Implement testing codes later.
## Finally
If you want to learn more about face recognition and face detection, this project([insightface](https://github.com/deepinsight/insightface)) may help you.
