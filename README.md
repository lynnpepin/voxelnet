# Voxelnet (defunct)


This is a project with the goal of re-implementing
[VoxelNet (Zhou, Tuzel, 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf).

This is not finished! This is an implementation of Voxelnet without CUDA dependencies. Just Tensorflow 2.0 Keras.

Specifically, the goal is that this project:

1. Implements the novel layers and loss described in the paper,
2. Builds a model from these layers,
3. Trains and evaluates against the relevant dataset,
4. Provides pre-trained weights.

## Motivation for this implementation

I had too much difficulty using other codesets and implementations. A lot of them share the same 'lineage' and so I wanted to start a new lineage that I found easier to use.

I wanted this implementation to:

1. Emphasize code-quality,
2. Use pure-Python + Tensorflow, minimizing extra dependencies,
3. Prefer functional approaches
4. Use reasonable default arguments to avoid parameter wrangling,
5. Modularity, so this works more like a "flat library" rather than a "deep system",
6. Parameter transparency: Avoid dicts-as-arguments. (No more `params` or `args` arguments!) 

So, we want our code to look like this:

## Architecture

- The `notebooks` provides examples and whatnot as I worked through implementation details. This includes unfinished parts and notes.
- The `library.py` provides every function and Keras layer we use in this project.
- The `model.py` provides an instantiation of the model using the layers in `library.py`.

## License

This is a WIP, and license is subject to change. 
