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
           Z
   Y      ↑
    ↖▄█▀▀▀▀▀▀▀▄
--  █▉       ☺ ▉
-- ██▄▄▄▄▄▄▄▄▄▄▄█   →X
--   ▀██▀   ▀██▀
```


## 2. Partitioning (TODO)

This step partitions points into voxels.

The voxel grid corresponds to a range `(D, H, W)`,
corresponding to axes `(z, y, x)`,
with equal sized grid partitions `(v_D, v_H, v_W)`,
leading to a total grid size of `(D/v_D, H/v_H, W/v_W)`
(also denoted `(D', H', W')`.

Different partitioning parameters are used for different tasks. All units are in meters.

### 2.1. Partitioning spec for Car Detection

| Axis               |  `Z`      | `Y`         |  `X`       |
| ------------------ | --------- | ----------- | ---------- |
| Range  `(D, H, W)` | `[-3, 1]` | `[-40, 40]` | `[0, 70.4]`|
| Voxel sizes        | `0.4`     | `0.2`       | `0.2`      |
| `(v_D, v_H, v_W)`  | | | |
| Grid size          | `10`      | `400`       | `352`      |
| `(D', H', W')      | | | |


### 2.2. Partitioning spec for Pedestrian and Cyclist detection
