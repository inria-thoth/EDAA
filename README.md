# Hyperspectral Image Unmixing (HSU)

---

## Introduction

This repository implements various unmixing methods (supervised and blind) on different datasets.

---

## Installation

This repository was developed using Ubuntu 20.04 LTS, Python 3.8.8 and MATLAB 2020b.

We recommend using `conda` to handle the Python distribution and `pip` to install the Python packages.

```
$ conda create --name hsu python=3.8
```

```
$ pip install -r requirements.txt
```

To install `matlab.engine`, first visit the official [webpage](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

  1. Locate `matlabroot` (e.g. `~/softwares/matlab-2020b/`)
  2. Create a `matlab` build directory (e.g. `~/matlab/`)
  3. Locate your current `conda` environment (e.g. `~/conda/envs/hsu`)

```
$ cd $matlabroot/extern/engines/python
$ conda activate hsu
$ python setup.py build -b ~/matlab install --prefix ~/conda/envs/hsu
```

---

## Getting started

To be done.
