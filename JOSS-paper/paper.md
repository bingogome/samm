---
title: 'SAMM (Segment Any Medical Model): A 3D Slicer Integration to SAM'
tags:
  - Python
  - Segment Anything
  - Medical Image Analysis
  - 3D Slicer
authors:
  - name: Yihao Liu
    orcid: 0000-0002-2654-9793
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Jiaming Zhang
    orcid: 0009-0002-6787-8590
    equal-contrib: true
    affiliation: 1
  - name: Zhangcong She
    equal-contrib: true
    affiliation: 2
  - name: Amir Kheradmand
    orcid: 0000-0002-3630-3949
    affiliation: 3
  - name: Mehran Armand
    orcid: 0000-0003-1028-8303
    corresponding: false
    affiliation: 1
affiliations:
 - name: Department of Computer Science, The Johns Hopkins University, Baltimore, MD, USA
   index: 1
 - name: Department of Mechanical Engineering, The Johns Hopkins University, Baltimore, MD, USA
   index: 2
 - name: Department of Neurology, The Johns Hopkins University, Baltimore, MD, USA
   index: 3
date: 13 May 2023
bibliography: paper.bib

---

# Summary
The Segment Anything Model (SAM) is a new image segmentation tool trained with the largest available segmentation dataset. The model has demonstrated that, with efficient prompting, it can create high-quality, generalized masks for image segmentation. However, the performance of the model on medical images requires further validation. To assist with the development, assessment, and application of SAM on medical images, we introduce Segment Any Medical Model (SAMM), an extension of SAM on 3D Slicer - an open-source image processing and visualization software extensively used by the medical imaging community. This open-source extension to 3D Slicer and its demonstrations are posted on GitHub (https://github.com/bingogome/samm). SAMM achieves 0.6-second latency of a complete cycle and can infer image masks in nearly real-time.

# Statement of Need
The advent of Large Language Models (LLM) has led to significant progress in image analysis with potential for future advancements. SAM [@kirillov2023segment] is a revolutionary foundation model for image segmentation and has already shown to have the capability of handling diverse segmentation tasks. SAM especially prevails in zero-shot domain generalization cases as compared with existing elaborate, fine-tuned models trained for specific domains. An important prospect for the application of SAM would be its adaptation to the complex task of segmenting medical images with significant inter-subject variations and low signal-to-noise ratio.  

Segmentation allows separation of different structures and tissues in medical images, which are then used to detect the region of interest or reconstruct multi-dimensional anatomical models [@sinha2020multi]. However, the existing AI-based segmentation methods do not fully bridge the domain gap among different imaging modalities, such as computed tomography (CT), magnetic resonance imaging (MRI), and ultrasound (US) [@wang2020deep]. In this context, domain gap refers to the difference in the data format between the source and target domain, which is relevent to the challenge of training AI systems to perform common analyses tasks across different image modalities without the need for a comprehensive dataset. Each image modality offers a distinct advantage in visualizing specific anatomical structures and their abnormalities. Domain gap introduces specific challenges to medical image processing and analysis. The challenges are further elevated by anatomical differences and deformities (e.g. tumor, bone fracture) that may exist among individuals. Therefore, a universal tool for medical image segmentation can handle all modalities as well as anatomical structures. Conventional machine learning (ML) and deep Learning (DL) techniques can potentially achieve this goal with their model trained on large and domain-specialized medical image datasets. To achieve this, however, the ML and DL techniques have to overcome a series of critical challenges including (but not limited to) data privacy, ethics, expenses, scalability, data integrity, and validation [@gao2023]. In contrast, SAM can perform a new and/or different task at inference time without being trained on the data collected from that task [@kirillov2023segment]. 
This feature makes SAM promising for segmenting multi-modality medical images with less effort. 

Despite the extensive usage of AI-based methodologies in medical image analysis, the use of foundation models within this field remains a largely unexplored area of research. However, migrating SAM to the medical image analysis field requires resolving the difference in coordinate systems and image structure between medical images and normal images. 3D Slicer [@pieper20043d], as an open source software for medical images, provides routines to read and write various file formats, manipulate 2D and 3D coordinate systems, and present a consistent user interface paradigm and visualization tool. Here we provide a unified framework incorporating 3D Slicer and SAM to perform medical image segmentation.

Figure 1 presents the overall architecture of SAMM, which consists of a SAM server with a pre-trained model loaded and an interactive prompt plugin for 3D Slicer (Slicer-IPP). Slicer-IPP first handles all the image slices with the built-in interfaces of 3D Slicer. Then, it processes all the images and subsequently maps the embeddings of the images in a binary array format in Random Access Memory (RAM) for efficient storage and retrieval. 
The SAM Server runs in parallel with Slicer-IPP and keeps monitoring the request sent from 3D Slicer. The server end hosts a local SAM, which employs an embedding strategy to incorporate image data fetched from RAM into the model. The image encoder in SAM uses an MAE pre-trained Vision Transformer (ViT) [@dosovitskiy2020image] to downscale the input images and detect (embed) the image features synchronously. The Slicer-IPP provides a prompt (typically, a prompt is a point placed on the image slice) to the end users to add/remove a selected region. The prompt points are transmitted to the prompt encoder of SAM, which subsequently generates a mask using the prompts and pre-loaded image embeddings. The image embedding process is followed by a segmentation inference step based on both the embedding features and user-defined prompts during runtime. Note that the image encoders run once per image, rather than per prompt, which allows the users to segment the same image multiple times with different prompts in real-time. Given that the initialization of image embedding occurs in advance, the subsequent mask generation process can be performed with small latency (see Section ). 

The Slicer-IPP facilitates the intuitive alignment of diverse coordinates associated with the same target. It can work out with the discrepancies between the RAS (right, anterior, superior) coordinate system, the IJK (slice ID) coordinate system, and the image pixel coordinate system by providing proper conversion functionalities. For instance, at an inference request, Slice-IPP converts the coordinates of RAS to IJK to identify the image ID and then transmits the ID along with the prompts, with its coordinates converted according to the same pattern as how the image are converted, to SAM Server. The mask generated by the inference step, on the other hand, is transformed from the pixel to RAS coordinate system.

Slicer-IPP

In general, the Slicer-IPP is composed of a data communication module, a prompt labeling module, and a visualization module. The data communication module accepts any data formats from different modalities and packs them as binary image files used by SAM. The Slicer-IPP and SAM Server are designed to run five parallel tasks denoted as send inference request (SND_INF), receive inference request (RCV_INF), complete SAM inference (CPL_INF), receive mask transmission (RCV_MSK), and apply mask (APL_MSK). The affiliation of tasks is shown in Figure 3. Slicer-IPP hosts SND_INF, RCV_MSK, and CPL_INF, while the server end hosts RCV_MSK and APL_MSK. Each task is an independent loop that is executed synchronously. They run with the effort-first principle, and in the Slicer-IPP, each loop is set to have a 60 ms gap to process other tasks, since 3D Slicer is a single-threaded software. A complete inference cycle starts from SND_INF and ends with APL_MSK. The time latency of one inference cycle is discussed in Section `\ref{sec:result}`.


To facilitate communication between 3D Slicer and external tools or services, the platform uses ZeroMQ (ZMQ) [@2013zeromq] messaging library and Numpy [@numpy] memory mapping. ZMQ is a lightweight messaging library that enables high-performance, asynchronous communication between distributed applications. In SAMM, ZMQ and Numpy are employed to transfer images, prompts, and requests between the Slicer-IPP and the SAM Server. The segmentation task is eventually accelerated by applying these two packages. This integration enables researchers to take advantage of SAM's cutting-edge segmentation capabilities within the familiar 3D Slicer platform, expanding the range of tasks that can be performed on images using this powerful software tool. The use of ZMQ and Numpy memory mapping also provides the flexibility to customize the communication protocol to fit the user's specific needs, further enhancing the versatility of the 3D Slicer platform. 
