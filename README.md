<!-- # Hyperspectral Image Unmixing (HSU) -->
# Entropic Gradient Archetypal Analysis for Blind Hyperspectral Unmixing

---

Official PyTorch implementation of the paper _Entropic Gradient Archetypal Analysis for Blind Hyperspectral Unmixing_

## Introduction

This repository mainly implements the entropic gradient archetypal analysis method for blind hyperspectral unmixing.

In addition, we include various unmixing methods (supervised and blind) on several standard real datasets.

---

## Installation

This repository was developed using Ubuntu 20.04 LTS, Python 3.8.8 and MATLAB 2020b.

We recommend using `conda` to handle the Python distribution and `pip` to install the Python packages.

```shell
conda create --name edaa --file conda.txt
```

Activate the new `conda` environment to install the Python packages and run the code:
```shell
conda activate edaa 
```

```shell
git clone https://github.com/inria-thoth/EDAA
cd EDAA && pip install -r requirements.txt
```

---

### Run matlab code in Python

It is possible to run the [NMF-QMV](https://github.com/LinaZhuang/NMF-QMV_demo) code inside Python to make sure that the reported results are consistent across methods.

To install `matlab.engine`, first visit the official [webpage](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

  1. Locate `matlabroot` (e.g. `~/softwares/matlab-2020b/`)
  2. Create a `matlab` build directory (e.g. `~/matlab/`)
  3. Locate your current `conda` environment (e.g. `~/conda/envs/hsu`)

```
cd $matlabroot/extern/engines/python
conda activate hsu
python setup.py build -b ~/matlab install --prefix ~/conda/envs/hsu
```

---

### Data download

The data can be downloaded from [http://pascal.inrialpes.fr/data2/azouaoui/data.zip](http://pascal.inrialpes.fr/data2/azouaoui/data.zip).

To extract it in the appropriate `./data` directory, simply run:

```
python -m utils.prepare_datasets
```

If for whatever reasons you were not able to use the script above:

1. Download the file directly from the [link](http://pascal.inrialpes.fr/data2/azouaoui/data.zip).
2. Create a new folder `./data` in the root folder of this repository.
3. Extract `data.zip` into `./data`.
4. You should have all 7 datasets directly available in the `./data` folder.

---

## Getting started

This repository uses [hydra](https://hydra.cc/) to seamlessly manage different configurations over the command line.

To run the `EDAA` model on the `Samson` dataset, use the following command:

```
python main.py dataset=Samson mode=blind model=BlindEDAA
```

The different datasets available are listed under `./hsi_unmixing/config/dataset`:

* Samson
* JasperRidge
* Urban{4,5,6}
* TinyAPEX
* WDC

To run the `EDAA` model on all datasets, use the following command:

```
python main.py dataset=Samson,JasperRidge,Urban4,Urban6,TinyAPEX,WDC mode=blind model=BlindEDAA --multirun
```

---

## Other methods

This repository implements other supervised and blind hyperspectral unmixing techniques:

* **Supervised methods**
  * FCLSU using `VCA` (Python implementation) and DecompSimplex (`DS`) from `SPAMS` .
  * FCLSU using `VCA` (Python implementation) and `FCLS` from `pysptools` (slower).

Command to run FCLSU using VCA+DS on Samson:
```
python main.py dataset=Samson mode=supervised torch=False model=DS initializer=VCA
```

* **Blind methods**
  * `NMF-QMV` using a Python wrapper on the Matlab implementation.
  * Archetypal Analysis (`AA`) from `SPAMS`.
  * Robust AA (`RAA`) from `SPAMS`.
  
Command to run `AA` on Samson:
```
python main.py dataset=Samson mode=blind torch=False model=AA
```
