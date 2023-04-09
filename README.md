# Segment Any Medical-Model (SAMM)

[The Laboratory for Computational Sensing and Robotics (LCSR)](https://lcsr.jhu.edu/), [Johns Hopkins University](https://www.jhu.edu/)

[Yihao Liu](https://yihao.one/), [Jeremy Zhang](https://jeremyzz830.github.io/), Zhangcong She



## Motivation

Accurate image segmentation is crucial for medical image analysis as it enables clinicians to extract meaningful information from the image. It also allows for the detection of subtle changes in the tissue or organ of interest, which is essential for monitoring disease progression and treatment response. 

Our project, **Segment Any Medical-Model** aims to develop an integration for [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) and [3D Slicer](https://www.slicer.org/) for future development and validation of the potentials of transferring Large Language Model to the medical image analysis field.



## Tutorial

### Installation

It's essential to have a clean virtual environment to avoid any potential conflicts. Therefore, you'd better to create a new environment for the other 

#### Install SAM

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```



#### Install 3D Slicer

Follow this [page](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html) to download a compatible version of 3D Slicer and install it in your local environment.


#### Install the Extension to 3D Slicer

The source code of the extension is contained in [samm](/samm).



### Demo

#### Notebook



#### Youtube

https://www.youtube.com/watch?v=vZK45noZVIA
