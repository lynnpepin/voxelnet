# Voxelnet implemented in Tensorflow 2.0

This is a project with the goal of re-implementing
[VoxelNet (Zhou, Tuzel, 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf).

Specifically, the goal is that this project:

1. Implements the novel layers and loss described in the paper,
2. Builds a model from these layers,
3. Trains and evaluates against KITTI dataset,
4. Provides pre-trained weights.

## Motivation for this implementation

Why yet-another-implementation of VoxelNet? This is because of a widespread
problem of code-quality.

So, secondary goals are to have these qualities:

1. Functional: Prioritize functional approaches, making code properly modular.
   (So, layers and models should be ready-to-import with no setup.)
2. Documentation: Provide sphinx-RTD docstrings for usability and maintainability.
   (So that each item is clear on how to use it, and so documentation can be built.)
3. Reasonable defaults: As far as possible, functions / methods should have reasonable default arguments.
   (To speed-up research and to implicitly provide useful examples.)
4. Library-focused: This code should be a "flat library" rather than a "deep system".
   (No arg-parsing, global state, etc. should have to worry you!)
5. Transparency: Avoid dicts-as-arguments.
   (Too many projects pass `config.py` or `params` throughout a script, making code re-use difficult.) 

So, we want our code to look like this:

## Architecture

TODO-- document structure of this project

## License

This is a WIP, and license is subject to change. 
