# Faster R-CNN:

Introduces an RPN for object detection in images. Rectangular regions. Think Yolonet.

The big deal here is that it uses GPU, not CPU.

# Mask R-CNN:

Performs segmentation and region proposal (bounding box).

Mask R-CNN extends Faster R-CNN by adding segmentation in each region. [Intro, right column.]

 - From the region, use a small FCN for pixel-to-pixel segmentation.

"Mask R-CNN extends Faster R-CNN by adding a branch for predicting semgentation masks on each region."

This is a very simple but effective addition, which they say several times in the paper.


# car detection for av: lidar and vision fusion approach through deep learning framework

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8202234

2017 paper: Pointcloud to bounding box, refined using image region proposal.

We are doing something different: Image for lane detection, lidar for bounding box. Lane detection lets us identify lane index.

Can not find source code.

# CS230 PointFusion:

https://github.com/malavikabindhi/CS230-PointFusion/

Similar to above. It's an undergad project done in 2018.

Fusion uses pointcloud + image --> bounding box and classification.

This is an undergraduate course project but it might be the most directly useful. Pretrained model not provided.


# Deep Continuous Fusion for Multi-Sensor 3D Object Detection

https://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf

Fuse image and lidar and more, encode to birds-eye-view (squashed z-axis) space, perform object detection in this space (bounding boxes).

This is one that might work for us.

Code is not released. There is one incomplete reimplementation: https://github.com/Chanuk-Yang/Deep_Continuous_Fusion_for_Multi-Sensor_3D_Object_Detection

The incomplete implementation has **no comments in the model architecture**, except:

1. Some code that is commented out
2. Pre-generated PyCharm IDE comments to help new programmers run the code.

The test.py code has 5 meaningful comments in 316 lines of code. The train.py has 0 in 126.
The loss has 9 in 271, most of which are in Korean. They translate nicely on Google.

# PointRCNN

This is the most promising of what we've seen so far.

https://github.com/sshaoshuai/PointRCNN

This is good, it seems to be lidar --> bounding box. Implementation is "still in progress" but has a pretrained model. README is extensive. 

Has some C++ and CUDA, concerning.

... Code is structured very similarly to other code I've seen before. Very lacking in documentation. 32k lines of code. :)

# Awesome Point Cloud Analysis
https://github.com/Yochengliu/awesome-point-cloud-analysis


Looking at htis list now. A lot of things here. From 2021 down, and skipping datasets and segmentation.

## 2020 TANet   
https://github.com/happinesslz/TANet
https://arxiv.org/pdf/1912.05163.pdf
    
consider this.... Built off PointPillars, had difficulty in the past.


## 2019 Object Recognition ensemble / survey

https://arxiv.org/abs/1904.08159

## 2019 survey

https://arxiv.org/abs/1912.12033

https://github.com/QingyongHu/SoTA-Point-Cloud


## 2019 Points to Parts

https://arxiv.org/pdf/1907.03670.pdf

https://github.com/sshaoshuai/PointCloudDet3D points to open-mmlab/OpenPCDet


## 2019 StarNet

https://arxiv.org/pdf/1908.11069v1.pdf

Promising, source should be in lingvo by now, check


## 2019 Point-Voxel CNN

https://arxiv.org/abs/1907.03739

Not sure what it does but look at it.

Oh Hey, Song Han from MIT. Not the same Song Han who worked with the late Nhuong Nguyen on AET-SGD?

https://arxiv.org/pdf/2112.13935.pdf


## 2019 BBox for segmentation

https://arxiv.org/abs/1906.01140

Similar to Mask R-CNN, makes bounding boxes and then does segmentation in that.


## 2019 3D backbone for 3D ObjDet

https://arxiv.org/abs/1901.08373

https://github.com/Benzlxs/tDBN

## 2019 3D Yolo

https://github.com/AI-liu/Complex-YOLO

https://arxiv.org/pdf/1904.07537.pdf

## ... skipped some here to jump to voxelnet and below


## 2018 Pixor

https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf

https://github.com/ankita-kalra/PIXOR

## 


















