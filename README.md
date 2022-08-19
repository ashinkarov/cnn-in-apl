# Convolutional Neural Networks in APL

This directory contain multiple implementations of the CNN for handwritten
image recognition that follow Zhang's design.

The code is located in `cnn.apl`, and the input files are located under `input`
directory.  Note that if you are cloning the repo for the first time, you'd have
to manually download the MNIST data.

In order to run the code, you'd need to have an operational Dyalog APL
interpreter.  We have tried versions 16 and 17:
```bash
$ cat cnn.apl | dyalog
```

## APL Paper

We explain many details in the following paper: https://dl.acm.org/doi/pdf/10.1145/3315454.3329960

## SaC Paper

We wrote another paper about SaC versions of the same CNN: https://dl.acm.org/doi/10.1145/3460944.3464312
The main motivation is to speed-up the building blocks that are required to define the network without loosing too much expressiveness.

The implementation of this CNN in SaC is here: https://github.com/SacBase/array-2021-supmaterial

There is an online video where the SaC paper is presented: https://www.pldi21.org/prerecorded_array.3.html


