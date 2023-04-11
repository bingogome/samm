# Segment Any Medical-Model (SAMM): A 3D Slicer integration to Meta's SAM.

[Laboratory of Biomechanical and Image Guided Surgical Systems](https://bigss.lcsr.jhu.edu/), [Johns Hopkins University](https://www.jhu.edu/)

[Yihao Liu](https://yihao.one/), [Jeremy Zhang](https://jeremyzz830.github.io/), Zhangcong She

## Demo

https://www.youtube.com/watch?v=vZK45noZVIA

## Motivation

Accurate image segmentation is crucial for medical image analysis as it enables clinicians to extract meaningful information from the image. It also allows for the detection of subtle changes in the tissue or organ of interest, which is essential for monitoring disease progression and treatment response. 

Our project, **Segment Any Medical-Model** aims to develop an integration for [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) and [3D Slicer](https://www.slicer.org/) for future development and validation of the potentials of transferring Large Language Model to the medical image analysis field.

## Installation and How-To-Use

### TLDR version

```bash
git clone git@github.com:bingogome/samm.git
conda create --name samm
conda activate samm
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Start 3D Slicer, in the Python Console:

```python
slicer.util.pip_install("pyyaml")
slicer.util.pip_install("pyzmq")
```

SD Slicer -> `Developer Tools` &rarr; `Extension Wizard`.

`Extension Tools` -> `Select Extension` -> import the samm/samm folder. 

Back to terminal, cd to samm (upper level)

Run ./samm-python-terminal/sam_server.py

If it throws an error missing "sam_vit_h_4b8939.pth", move segment-anything/notebooks/sam_vit_h_4b8939.pth to samm/samm-python-terminal/samm-workspace

Follow the [demo](https://www.youtube.com/watch?v=vZK45noZVIA) and Segment Any Medical Model away!


### Install samm

Install this repo:

```bash
git clone git@github.com:bingogome/samm.git
```

### Create Virtual Environment

It's essential to have a clean virtual environment to avoid any potential conflicts. Therefore, you'd better to create a new environment for running the rest part of the code.

Install any version of anaconda to manage the virtual environment. Anaconda installation guide can be found [here](https://docs.anaconda.com/anaconda/install/).

Create virtual environment and activate it:

```bash
conda create --name samm
conda activate samm
```

Note: The given python script in this [folder](/samm-python-terminal) has to be executed in samm venv.

### Install SAM

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

Then,

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```


### Install 3D Slicer

Follow this [page](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html) to download a compatible version of 3D Slicer and install it in your local environment.


### Install the SAMM Extension to 3D Slicer

The source code of the extension is contained in [samm](/samm).

In the GUI of 3D Slicer, expand the extension drop-down menu, and choose `Developer Tools` &rarr; `Extension Wizard`.

Then on the left side of the GUI, click the toggle bar named `Extension Tools` and click `Select Extension' button. It will prompt a navigation window where you can find, select and import the samm folder. 

## Tutorial

To launch the extension, go back to the extension drop-down menu, and choose `Segmentation`  &rarr;`Samm Base`. 




