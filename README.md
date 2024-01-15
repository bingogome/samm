# Segment Any Medical-Model (SAMM): A 3D Slicer integration to Meta's SAM.

[paper](https://arxiv.org/abs/2304.05622)
\
[Laboratory of Biomechanical and Image Guided Surgical Systems](https://bigss.lcsr.jhu.edu/), [Johns Hopkins University](https://www.jhu.edu/)
\
[![The Video](https://github.com/bingogome/samm/blob/main/thumbnail.png)](https://youtu.be/tZRG7JljEBU)
[![The Video](https://github.com/bingogome/samm/blob/main/thumbnail2.png)](https://youtu.be/tZRG7JljEBU =x800)

# Table of contents
- [Segment Any Medical-Model (SAMM): A 3D Slicer integration to Meta's SAM.](#segment-any-medical-model-samm-a-3d-slicer-integration-to-metas-sam)
- [Table of contents](#table-of-contents)
  - [Introduction ](#introduction-)
  - [Before You Try ](#before-you-try-)
  - [How to Use ](#how-to-use-)
    - [Features ](#features-)
    - [Add Your Own SAM Vriant ](#add-your-own-sam-vriant-)
  - [Installation Guide ](#installation-guide-)
    - [Prerequisite ](#prerequisite-)
    - [TLDR version](#tldr-version)
  - [Citation ](#citation-)

## Introduction <a name="introduction"></a>
What are SAM, SAMM and SAMME?
* SAM is the vision foundation model developed by Meta, [Segment Anything](https://segment-anything.com).
* SAMM is an engineering integration of SAM to 3D Slicer, intended for medical image segmentation. The name is the abbreviation of Segment Any Medical Model.
* SAMME is an extended version of SAMM supporting not only the vanilla SAM, but new variants from the community.
\
Why SAMM and SAMME?
* SAMM was a side project for fun initially (you can tell from the name). Later we got some interests from people because hey it's a new model and it's cool. Accurate image segmentation is crucial for medical image analysis as it enables clinicians to extract meaningful information from the image. It also allows for the detection of subtle changes in the tissue or organ of interest, which is essential for monitoring disease progression and treatment response. 
* This later became aiming to develop an integration for [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) and [3D Slicer](https://www.slicer.org/) for future development and validation of the potentials of transferring Foundation Model to the medical image analysis field. [More and more variants](https://github.com/YichiZhang98/SAM4MIS) of SAM emerged, so we thought it's probably worth it to have a platform supporting the addition of new SAMs.
\

## Before You Try <a name="before-you-try"></a>
Make sure you have more than 8GB of VRAM so that it doesn't crash. \

## How to Use <a name="how-to-use"></a>
Also see the [Installation Guide](#installation-guide). 
\
Watch this [video](https://www.youtube.com/watch?v=tZRG7JljEBU). Here are some key points demonstrated in the video.
* 00:00 Start Server
* 00:20 Pick a Model
* 00:22 Calculate Embeddings
* 00:37 3D Bounding Box
* 01:20 Switch Models
* 01:41 Fix Bad Results by Adjust Brightness and Contrast
* 02:33 2D Bounding Box
* 02:45 Prompt Propagation
* 03:05 Change Prompting View
* 03:27 Point Prompts

### Features <a name="features"></a>
- 3 View Inference
- Embedding saving
- Data type
  - Volume
  - 2D Image
  - RGB Image (WIP)
- models
  - vit_b
  - vit_h
  - vit_l
  - vit_t - [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
  - vit_b - [MedSAM](https://github.com/bowang-lab/MedSAM)
- interactions
  - positive and negative points
  - 2D bounding box
  - 3D bounding box 
  - combination
  - automatic segmentation
- training (WIP)

### Add Your Own SAM Vriant <a name="add-your-own-sam-variant"></a>
The inference can run in real time (on a 3090) once the embeddings of the images are calculated and loaded. If you'd like to add your own SAM variant, make sure the implementation keeps the Predictor class intact or uses the same interface to call. Make sure the architecture follows the similar component make ups as in the vanilla SAM.

## Installation Guide <a name="installation-guide"></a>
Works both on Linux and Windows. Has Mac support, but not tested yet.

### Prerequisite <a name="prerequisite"></a>
This assumes Cuda, cv2 and pytorch are in your environment.

### TLDR version
```bash
git clone git@github.com:bingogome/samm.git
conda create --name samm
conda activate samm
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx timm
git clone https://github.com/bowang-lab/MedSAM
pip install -e MedSAM
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```
If you are using Windows, it's okay if you don't install pycocotools.

```bash
cd samm/samm-python-terminal
mkdir samm-workspace
```
Then, move the check point files in the `samm-workspace` folder.

Start 3D Slicer, in the Python Console:

```python
slicer.util.pip_install("pyyaml")
slicer.util.pip_install("pyzmq")
slicer.util.pip_install("tqdm")
```

`SD Slicer` &rarr; `Developer Tools` &rarr; `Extension Wizard`.

`Extension Tools` &rarr; `Select Extension` &rarr; import the samm/samm folder. 

Back to terminal, cd to root folder `samm`

```bash
python ./samm-python-terminal/sam_server.py
``` 

If it throws an error missing "sam_vit_h_4b8939.pth", move segment-anything/notebooks/sam_vit_h_4b8939.pth to samm/samm-python-terminal/samm-workspace

Follow the [demo](https://www.youtube.com/watch?v=tZRG7JljEBU) and Segment Any Medical Model away!

## Citation <a name="citation"></a>
If you use SAMM in your research, please consider use the following BibTeX entry.

```bibtex
@article{liu2023samm,
  title={SAMM (Segment Any Medical Model): A 3D Slicer Integration to SAM},
  author={Liu, Yihao and Zhang, Jiaming and She, Zhangcong and Kheradmand, Amir and Armand, Mehran},
  journal={arXiv preprint arXiv:2304.05622},
  year={2023}
}
```

