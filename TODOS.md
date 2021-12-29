# Big List Of TODOS


- [ ] Data loading:
    - [ ] KITTI data tools and prep
    - [ ] Data augmentations for training:
        - [ ] Perturbation
        - [ ] Global scaling
        - [ ] Global rotation
- [~] Feature learning network:
    - [x] Partitioning points
    - [x] Grouping points into voxelgrid
    - [x] Sampling points within voxels
        - [~] (Sparse tensor representation starts here)
    - [x] Augmenting voxel-cloud with offsets
    - [x] VFE-n custom layers
        - [x] Fully-connected component
        - [x] Elementwise maxpool
        - [x] Pointwise concatenation 
        - [x] VFE-n custom layer
    - [x] VFE-out custom layer
- [x] Convolutional middle layers:
    - [x] Convolutional middle block custom layer
    - [x] Convolutional middle layers
- [x] Region proposal network:
    - [x] RPN Convolutional Block custom layer
    - [x] RPN network
- [~] Loss function:
    - [ ] SmoothL1 loss function
    - [ ] Loss anchors
    - [ ] Loss custom layer
- [ ] Training
- [ ] Optimizations
    - [~] Sparse tensor representation
    - [ ] Sparse operations over tensors
- [~] Fit CARLA lidar to KITTI specifications
- [~] Integrate into CARLA


Progress since last week:

Finished:
    - Partitioning and grouping code
    - Sampling code (naive sparse)
    - Augment-in-sample
    - VFE FCN, maxpool layer
    - VFE-n neural network layer (tentative)
    - VFE-out neural network layer (tentative)

Work on approach:
    - Sparse tensor representation with dense tensor