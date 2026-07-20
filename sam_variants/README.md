# SAMM Model Variants

This folder holds upstream model repositories cloned by SAMM setup tasks. The
cloned repositories and model checkpoints are generated local files and are not
committed.

## Setup Commands

Run these commands from a terminal before using the corresponding backend in
Slicer. They install upstream code, Pixi environments, and checkpoints.

From the project root, set up every backend:

```bash
pixi run setup-variants
```

Or set up one variant when you only need a subset in Slicer:

```bash
pixi run setup-sam1
pixi run setup-sam2
pixi run setup-mobile-sam
pixi run setup-medsam
pixi run setup-medsam2
pixi run setup-sam3
pixi run setup-medical-sam3
pixi run setup-fastsam
```

Each task clones the upstream repository when it is missing, prepares public or
linked checkpoints when possible, installs the Pixi environment, editable-installs
the cloned package when that backend needs one, runs `pip check`, and runs an
import smoke check. Existing clones are reused as-is. After setup completes,
switch back to Slicer for preparing models, prediction, embeddings, and
finetuning UI workflows.

Review and update a clone explicitly when you want a newer upstream revision.

## Cloned Repositories

| Variant | Local folder | Pixi environment |
| --- | --- | --- |
| SAM 1 | `sam_variants/segment-anything` | `sam1` |
| SAM 2 | `sam_variants/sam2` | `sam2` |
| MobileSAM | `sam_variants/MobileSAM` | `mobile-sam` |
| MedSAM | `sam_variants/MedSAM` | `medsam` |
| MedSAM2 | `sam_variants/MedSAM2` | `medsam2` |
| SAM 3 | `sam_variants/sam3` | `sam3` |
| Medical-SAM3 | `sam_variants/Medical-SAM3` | `medical-sam3` |
| FastSAM | `sam_variants/FastSAM` | `fastsam` |

## Checkpoints

All user-facing checkpoint paths live under `checkpoints/`.

| Variant | Files |
| --- | --- |
| SAM 1 | `sam_vit_b_01ec64.pth`, `sam_vit_l_0b3195.pth`, `sam_vit_h_4b8939.pth` |
| SAM 2.1 | `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_small.pt`, `sam2.1_hiera_base_plus.pt`, `sam2.1_hiera_large.pt` |
| MobileSAM | `mobile_sam.pt` |
| MedSAM | `medsam_vit_b.pth`, `medsam_text_prompt_flare22.pth` |
| MedSAM2 | `MedSAM2_latest.pt`, `MedSAM2_2411.pt`, `MedSAM2_CTLesion.pt`, `MedSAM2_MRI_LiverLesion.pt`, `MedSAM2_US_Heart.pt` |
| SAM 3 | `sam3.pt` |
| Medical-SAM3 | `medical_sam3.pt` |
| FastSAM | `FastSAM-x.pt`, `FastSAM-s.pt` |

Setup behavior:

- SAM 1, SAM 2.1, and MedSAM2 checkpoints are downloaded automatically when missing.
- MobileSAM ships `weights/mobile_sam.pt` in the upstream repository; setup links it into `checkpoints/`.
- MedSAM base and MedSAM Text checkpoints are Google Drive downloads. Setup prints red missing-checkpoint instructions and accepts the files in `checkpoints/`.
- SAM 3 is gated on Hugging Face. Get approval at <https://huggingface.co/facebook/sam3>, then run `pixi run setup-sam3`; setup asks for confirmation, runs `hf auth login`, and downloads `sam3.pt`.
- Medical-SAM3 setup clones <https://github.com/AIM-Research-Lab/Medical-SAM3>, installs its bundled `sam3` package in the `medical-sam3` environment, and uses `hf download` to fetch the upstream 2D checkpoint from <https://huggingface.co/ChongCong/Medical-SAM3> as `checkpoints/medical_sam3.pt`.
- FastSAM checkpoints are Google Drive downloads. Setup prints the direct download pages and accepts `FastSAM-x.pt` and/or `FastSAM-s.pt` in `checkpoints/`.

## Prompt Support

Capabilities are registered in `service/samm_server/model_registry.py` and are
returned by `GET /models`.

| Variant | Points | 2D box | 3D box | Mask input | Text | Auto 2D | Cached embeddings | Temporal API |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 1 | Yes | Yes | Yes | Yes | No | Yes | Yes | No |
| SAM 2 | Yes | Yes | Yes | Yes | No | Yes | Yes | Yes |
| MobileSAM | Yes | Yes | Yes | Yes | No | Yes | Yes | No |
| MedSAM | No | Yes | Yes | No | No | Yes | Yes | No |
| MedSAM Text | No | No | No | No | Yes | Yes | Yes | No |
| MedSAM2 | Yes | Yes | Yes | Yes | No | Yes | Yes | Yes |
| SAM 3 | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Medical-SAM3 | No | Yes | Yes | No | Yes | Yes | Yes | No |
| FastSAM | Yes | Yes | Yes | No | Yes | Yes | No | No |

The table describes capabilities exposed through SAMM, not every experimental
entry point that may exist in an upstream repository. The service returns these
flags for each weight, and Slicer uses them to enable prompts, embedding
actions, scrolling auto prediction, and slices-as-video buttons.

- SAM 1, SAM 2.1, MobileSAM, and MedSAM2 support the usual compatible point, box, and mask combinations for 2D prediction.
- MedSAM base requires a box; MedSAM Text accepts text only.
- SAM 3 supports text alone or text+box for grounding, and points, box+points, or mask through its interactive path. Text is not combined with points or mask input.
- Medical-SAM3 supports box, text, and SAMM's processor-backed text+box combination, but not point or mask input with the released 2D checkpoint.
- FastSAM accepts exactly one of text, points, or box for each prediction.

The existing Slicer 3D-box and scrolling auto-predict controls retain their 2D
paths. SAM 2, MedSAM2, and SAM 3 additionally expose `Predict 3D box (Slices as
video)` with a middle-frame box seed and `Auto predict (Slices as video)` with
the current prompted slice as its seed. They use the protocol 1.2 temporal API
and propagate in both directions. SAM 2 and MedSAM2 accept point, box, or mask
seeds. SAM 3 accepts text, box, text+box, points, or points+box; its public video
API does not expose mask seeding.

The normal `Predict 3D box` mode projects the ROI to each intersecting slice and
runs independent 2D predictions for every backend with 3D-box capability. The
slices-as-video variant uploads the ordered stack, seeds the middle ROI frame,
propagates in both directions, streams cursor-polled results, and clips masks to
the original ROI. `Auto predict (Slices as video)` instead uploads the complete
selected view and seeds the current slice without ROI clipping. Neither video
mode requires cached 2D embeddings.

## Variant Notes

Academic references for the upstream models are collected in
[Citations](#citations).

### SAM 1

SAM 1 uses the upstream `SamPredictor` interface and supports the standard SAM
prompt set: points, boxes, and mask input. Embeddings can be cached, saved, and
loaded through SAMM embedding folders.

### SAM 2

SAM 2 prepares `SAM2VideoPredictor` and wraps that same model with
`SAM2ImagePredictor`, preserving static prediction without loading a second
checkpoint copy. Protocol 1.2 exposes lossless in-memory slice stacks through
temporal jobs with point, box, or mask conditioning and forward/backward
propagation.

### MobileSAM

MobileSAM follows the SAM 1 predictor path with the ViT-T weight.

### MedSAM

MedSAM base is box-only. Use it with a 2D box, 3D box, or auto prediction with a
box prompt. The MedSAM base checkpoint page is:

<https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link>

MedSAM Text FLARE22 is a separate text-only backend. It uses a CLIP text prompt
encoder and the FLARE22 text prompt checkpoint:

<https://drive.google.com/file/d/12YH-N6PAKayulhS99MBURVNpuQtVj98S/view?usp=sharing>

### MedSAM2

MedSAM2 reuses SAMM's shared SAM2 image/video backend against the MedSAM2
package and `configs/sam2.1_hiera_t512.yaml`. Its native NPZ predictor accepts
the preprocessed slice tensor directly for temporal jobs. `MedSAM2_latest.pt`
is the first listed weight. Finetuned runs that write
`finetuning_runs/<run>/samm_model.json` appear as additional MedSAM2 weights.

### SAM 3

SAM 3 uses `Sam3Processor` for text and text+box grounding. Points, box+points,
and mask input use SAM3's interactive predictor. Text mixed with points or mask
input is rejected by the backend; text alone and text+box are supported. For
temporal jobs, SAMM builds the upstream unified video model once and reuses its
detector for image prediction, avoiding a second checkpoint copy. Slice frames
stay in memory without a JPEG-folder round trip. Video output objects are
unioned into SAMM's single output mask. Temporal mask seeds remain disabled
because the upstream public video prompt API does not expose them.

### Medical-SAM3

Medical-SAM3 is cloned from AIM Research Lab's upstream repository. The repo
bundles its own `sam3/` package, so setup installs that clone into the separate
`medical-sam3` environment instead of sharing the SAM 3 environment. Its 2D
checkpoint supports box and text grounding but does not include SAM3's separate
point/mask interactive-head weights. Upstream's 2D evaluation exercises box and
text separately; SAMM also exposes the processor-supported text+box combination.
SAMM therefore disables point and mask prompts for this variant while retaining
auto 2D and cached embeddings. Setup runs
`hf download ChongCong/Medical-SAM3 checkpoint_2D.pt` and stores it as
`checkpoints/medical_sam3.pt`; the upstream `checkpoint_3D.pt` is for its 3D
training/evaluation workflow and is not used by SAMM's image worker.

### FastSAM

FastSAM generates mask proposals for each direct prediction and filters them
with exactly one prompt mode: text, points, or box. It does not support mask
input or cached embeddings in SAMM, so auto prediction runs direct image
prediction and does not require `Embed view`.

FastSAM's upstream examples may refer to the x checkpoint as `FastSAM.pt`; SAMM
uses the downloaded filename `FastSAM-x.pt`.

## Citations

If you use a model through SAMM, cite the corresponding upstream work.

### SAM 1

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

### SAM 2

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

### MobileSAM

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
}
```

### MedSAM

```bibtex
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={654},
  year={2024}
}
```

### MedSAM2

```bibtex
@article{MedSAM2,
  title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
  author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
  journal={arXiv preprint arXiv:2504.03600},
  year={2025}
}
```

### SAM 3

```bibtex
@misc{carion2025sam3segmentconcepts,
  title={SAM 3: Segment Anything with Concepts},
  author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
  year={2025},
  eprint={2511.16719},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.16719}
}
```

### Medical-SAM3

```bibtex
@article{jiang2026medicalsam3,
  title={Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation},
  author={Jiang, Chongcong and Ding, Tianxingjian and Song, Chuhan and Tu, Jiachen and Yan, Ziyang and Shao, Yihua and Wang, Zhenyi and Shang, Yuzhang and Han, Tianyu and Tian, Yu},
  journal={arXiv preprint arXiv:2601.10880},
  year={2026},
  url={https://arxiv.org/abs/2601.10880}
}
```

### FastSAM

```bibtex
@misc{zhao2023fast,
  title={Fast Segment Anything},
  author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
  year={2023},
  eprint={2306.12156},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Related Documentation

- [../README.md](../README.md): Slicer setup and complete user workflow.
- [../service/README.md](../service/README.md): capability payloads and prediction APIs.
- [../finetuning/README.md](../finetuning/README.md): MedSAM2 dataset and finetuning workflow.
