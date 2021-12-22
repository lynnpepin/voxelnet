# Voxelnet Implementations

This is a list of Voxelnet Implementations we looked at and couldn't use.

Sorting by forks. (A flawed metric but the best we have.)

## 1. https://github.com/traveller59/second.pytorch


 - Relies on spconv (which we've had difficulty with in the past)
 - uses code from traveller59 (which we've had difficulty with in the past)
    - In terms of use, debugging, and documentation.
 - Several known bugs and flaws

**Did not use: Prior insurmountable difficulties**

## 2. https://github.com/qianguih/voxelnet

 - Tried this repo
 - Considered WIP, last update 4 years ago

**Did not use: Incompatible with CUDA version.**

## 3. https://github.com/tsinghua-rll/VoxelNet-tensorflow

 - Good because it's not a pure Python implementation
 - Bad because no results 

**Did not use: Incompatible with CUDA version.**

## 4. https://github.com/skyhehe123/VoxelNet-pytorch

**Did not use: PyTorch 0.3**

## 5. https://github.com/AbangLZU/VoxelNetRos

**Did not use: Made for ROS environment.**

## 6. https://github.com/steph1793/Voxelnet

- Requires C++ module
- Good, tflow 2.0
- Requires `kitti_eval` folder that does not exist; obviously incomplete.
    - Readme also has bugs in install code. (`setup` instead of `-m setup` or `setup.py`)

**Did not use: Instructions incomplete**

## 7. https://github.com/pyun-ram/FL3D

**Did not use: Not VoxelNet, no install directions**

## 8. https://github.com/collector-m/VoxelNet_CVPR_2018_PointCloud

**Did not use: PyTorch < 1.0**

## 9. https://github.com/edward0im/voxelnet_ros

**Did not use: ROS**

## 10. https://github.com/345ishaan/DenseLidarNet

**Did not use: Not VoxelNet, no install instructions**

## 11. https://github.com/maudzung/CenterNet3D-PyTorch

**Did not use, not VoxelNet**

## 12. https://github.com/baudm/VoxelNet-Keras

**Did not use, incomplete, no install instructions, old.**

## 13. https://github.com/yukitsuji/voxelnet_chainer

**Did not use, uses Chainer (unfamiliar), no install instructions, no requirements.txt, etc.**

## 14. https://github.com/fschaeffler93/best_voxelnet_ever

- Promising name
- No requirements.txt, but has install instructions
- Does not list Tensorflow version!!

**Did not use: Incomplete install instructions.**

---

It was at this point **that I instead search for vanilla VoxelNet implementations updated within the past two years, with >50 stars.**

## 15. https://github.com/fikki-maul/VoxelNet


