# ENSTRECT: Enhanced Structural Inspection

Enstrect is a pipeline to perform 2.5D instance segmentation of structural damages through *image-level segmentation/detection*, *prediction mapping* from 2D to 3D, and *damage extraction* (centerline and bounding polygon).

![Enstrect](https://github.com/user-attachments/assets/94b79295-d3c4-4101-9441-f69dbb8a6ec2)


## Citation
[ArXiV Preprint](https://arxiv.org/abs/2401.03298)

If you find our work useful, kindly cite accordingly:
```
@InProceedings{benz2024enstrect,
    author       = {Benz, Christian and Rodehorst, Volker},
    title        = {ENSTRECT: A Stage-based Approach to 2.5D Structural Damage Detection},
    booktitle    = {Computer Vision -- ECCV 2024 Workshops},
    publisher    = {Springer},
    city         = {Cham},
    year         = {2024}
}
```

## Installation
### General
```
git clone https://github.com/ben-z-original/enstrect.git
conda create --name enstrect python=3.10
conda activate enstrect
pip install -e .
```

### PyTorch3D Issues
Note that the PyTorch3D installation is known to be tricky. This especially refers to the alignment of versions of CUDA, PyTorch, and PyTorch3D. The configuration that worked in the project is:
- Ubuntu 20.04
- CUDA 12.1 with NVIDIA GeForce RTX 2080 Ti
- PyTorch 2.4.0
- Pytorch3D 0.7.7

## Data
The four segments (two for Bridge B and two for Bridge G) are provided in the assets. 
Note that they are part of a bigger dataset project, which will probably/hopefully released some day soon.

The dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1a1zwuuvaDVfmovGbcEsazfxr7OLfusLM/view?usp=sharing). 


For running the example, it must be corretly placed in the ```assets``` folder in the repository tree:
``` bash
└── enstrect
    ├── ...
    └── src
        └── enstrect
            ├── ...
            └── assets
                ├── example_image.png
                └── segments  # <-here it goes (unzipped)
```

## Run Example
### Cracks


### Exposed Rebar

## Segmentation Model
Three models for structural damage/crack segmentation were investigated in this work. 
The first is shipped with this repo, the other can be found in the respective repos:
- **nnU-Net-S2DS**: provided in this repo [[Path](./src/enstrect/segmentation/nnunet_s2ds.py)]
- **TopoCrack**: refer to repo [[Link](https://github.com/eesd-epfl/topo_crack_detection)]
- **DetectionHMA**: refer to repo [[Link](https://github.com/ben-z-original/detectionhma)]

Any other segmentation model can be used after being correctly wrapped into the SegmenterInterface, 
see here [[Path](./src/enstrect/segmentation/base.py)].


