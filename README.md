# ENSTRECT: A Pipeline for Enhanced Structural Inspection

ENSTRECT – short for *<ins>En</ins>hanced <ins>Str</ins>uctural Insp<ins>ect</ins>ion* – represents a workflow for 2.5D instance segmentation of *structural damages* (crack, corrosion, spalling, and exposed rebar) through *image-level segmentation/detection*, *prediction mapping* from 2D to 3D, and *damage extraction* (centerline and bounding polygon).

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
#### General
```
git clone https://github.com/ben-z-original/enstrect.git
conda create --name enstrect python=3.10
conda activate enstrect
pip install -e .
```

#### PyTorch3D Issues
Note that the PyTorch3D installation is known to be tricky. This especially refers to the alignment of versions of CUDA, PyTorch, and PyTorch3D. The configuration that worked in the project is:
- Ubuntu 20.04
- CUDA 12.1 with NVIDIA GeForce RTX 2080 Ti
- PyTorch 2.4.0
- Pytorch3D 0.7.7

## Data
The four segments (two for *Bridge B* and two for *Bridge G*) are provided in the assets. 
Note that they are part of a bigger dataset project, which hopefully will be released some day soon.

#### Download
The segments can be downloaded
- by running ```python -m enstrect.datasets.download``` (which places the segments correctly in the repo tree) 
- or from [Google Drive](https://drive.google.com/file/d/1QkyoZ1o9uKuxpLIlSZ-iA9jcba46oIwW/view?usp=sharing) (correct placement in repo tree required, see below).

#### Sample Points
Due to the overly large size of the point clouds for higher object resolutions (one message of the publication), only the reduced point clouds (```pcd_reduced.ply```) are shipped with the download. If you need the point clouds corresponding to the best image resolution, run:
```python -m enstrect.datasets.sample_points``` for the Bridge B, test segment. For information about the right command line arguments ```python -m enstrect.datasets.sample_points --help```.

## Segmentation Model
Three models for structural damage/crack segmentation were investigated in this work. 
The first one is shipped with this repo, the others can be found in the respective repos:
- **nnU-Net-S2DS**: provided in this repo [[repo internal path](./src/enstrect/segmentation/nnunet_s2ds.py)]
- **TopoCrack**: refer to repo [[Link](https://github.com/eesd-epfl/topo_crack_detection)]
- **DetectionHMA**: refer to repo [[Link](https://github.com/ben-z-original/detectionhma)]

The *nnU-Net-S2DS* segmentation model is supposed to be downloaded automatically when running the example. Otherwise it can be found here [Google Drive](https://drive.google.com/file/d/1UeXzpH76GYtZtyn2IjhDvD5Qu3u91YcC/view?usp=sharing) and needs to be placed under ```src/enstrect/segmentation/checkpoints```.

Any other segmentation model can be used with ENSTRECT after being correctly wrapped into the ```SegmenterInterface```, 
see here [[repo internal path](./src/enstrect/segmentation/base.py)].

## Run Example
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

### Cracks


### Exposed Rebar




