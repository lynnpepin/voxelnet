# VoxelNet

This specification is based off the VoxelNet paper:

> ["VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection" by Yin Zhou and Oncel Tunzel](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf)
> 
> ```
@inproceedings{zhou2018voxelnet,
  title={Voxelnet: End-to-end learning for point cloud based 3d object detection},
  author={Zhou, Yin and Tuzel, Oncel},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4490--4499},
  year={2018}
}
```

But this is an academic paper. So, let's note the details of VoxelNet, arranging the details layer-by-layer.

## 1. Point-Cloud Input

The input is a "point cloud" obtained by Lidar.

1. Input shape `(n, 4)`
2. Each point is `(x, y, z, r)`
3. The axes are:
	- `x` is `forward-facing`
	- `y` is `left-facing`
	- `z` is `upward-facing`

That is, in ASCII diagram form,

```
           Z (D)
   Y(H)    ↑
    ↖▄█▀▀▀▀▀▀▀▄
--  █▉       ☺ ▉
-- ██▄▄▄▄▄▄▄▄▄▄▄█   →X(W)
--   ▀██▀   ▀██▀
```


## 2. Partitioning (TODO)

This step partitions points into the voxel grid.

The voxel grid corresponds to a range `(D, H, W)`, corresponding to axes `(z, y, x)`,
with equal sized grid partitions `(v_D, v_H, v_W)`, leading to a total grid size of `(D/v_D, H/v_H, W/v_W)`,
(also denoted `(D', H', W')`).

Different partitioning parameters are used for different tasks. All units are in meters.

### 2.1. Partitioning spec for Car Detection

| Axis                           | `Z (D)`    | `Y (H)`     | `X (W)`    |
| ------------------------------ | ---------- | ----------- | ---------- |
| Range  `(D, H, W)` 			 | `[-3, 1]`  | `[-40, 40]` | `[0, 70.4]`|
| Voxel sizes `(v_D, v_H, v_W)`  | `0.4`      | `0.2`       | `0.2`      |
| Grid size  `(D', H', W')`      | `10`       | `400`       | `352`      |


They use maximum-sampling count `T = 35` per voxel.

The network is then as follows:

 1. Two VFE layers:
 	1. `VFE-1(7,32)`
 	2. `VFE-2(32,128)`
 	3. Final FCN maps VFE-2 to `R^{128}`.
 2. So, feature learning set generates sparse tensor with shape `(128, 10, 400, 352)`.
 3. Then these layers:
 	1. `Conv3D(128, 64, 3, (2, 1, 1), (1, 1, 1))`
 	2. `Conv3D( 64, 64, 3, (1, 1, 1), (0, 1, 1))`
 	3. `Conv3D( 64, 64, 3, (2, 1, 1), (1, 1, 1))`
 4. Yielding a 4D tensor of size `(64, 2, 400, 352)`.
 5. Reshape to feature map of size `(128, 400, 352)`, corresponding to channel, height, width.

Anchor sizes:

- `l^a = 3.9`
- `w^a = 1.6`
- `h^a = 1.56`

Centered at `z^a_c = -1.0`, with two rotations, `0` and `90` degrees. (See Fig 4.)


An anchor is considered "positive" if it beats intersection-over-union (IoU) with ground truth,
or if birds-eye view IoU is `> 0.60`. Anchors are "don't care" if `0.45 <= IoU <= 0.6`.



The loss function (see below) uses `a = 1.5` and `b = 1`.


### 2.2. Partitioning spec for Pedestrian and Cyclist detection

| Axis                           | `Z (D)`    | `Y (H)`     | `X (W)`    |
| ------------------------------ | ---------- | ----------- | ---------- |
| Range  `(D, H, W)` 			 | `[-3, 1]`  | `[-20, 20]` | `[0, 70.4]`|
| Voxel sizes `(v_D, v_H, v_W)`  | `0.4`      | `0.2`       | `0.2`      |
| Grid size  `(D', H', W')`      | `10`       | `200`       | `240`      |
