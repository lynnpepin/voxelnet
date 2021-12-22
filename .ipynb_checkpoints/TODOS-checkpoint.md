# Big List Of TODOS


- [ ] Data loading:
    - [ ] KITTI data tools and prep
    - [ ] Data augmentations for training:
        - [ ] Perturbation
        - [ ] Global scaling
        - [ ] Global rotation
- [ ] Feature learning network:
    - [x] Partitioning points
    - [x] Grouping points into voxelgrid
    - [x] Sampling points within voxels
        - [ ] (Sparse tensor representation starts here)
    - [x] Augmenting voxel-cloud with offsets
    - [ ] VFE-n custom layers
        - [x] Fully-connected component
        - [ ] Elementwise maxpool
        - [ ] Pointwise concatenation 
        - [ ] VFE custom layer
    - [ ] VFE-out custom layer
- [x] Convolutional middle layers:
    - [x] Convolutional middle block custom layer
    - [x] Convolutional middle layers
- [x] Region proposal network:
    - [x] RPN Convolutional Block custom layer
    - [x] RPN network
- [ ] Loss function:
    - [ ] SmoothL1 loss function
    - [ ] Loss anchors
    - [ ] Loss custom layer
- [ ] Training
- [ ] Optimizations
    - [ ] Sparse tensor representation
    - [ ] Sparse operations over tensors
- [ ] Fit CARLA lidar to Nuscenes specifications
- [ ] Integrate into CARLA


Progress since last week:

Finished:
    - partitioning
    - grouping
    - sampling
    - augmenting

Progressed on:
    - Sparse tensor representation
    - Understanding of remaining tasks
Understanding of:
    - VFE
    - Sparse representation as dense tenxor