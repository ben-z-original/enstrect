# ENSTECT: Enhanced Structural Inspection


## Citation
[ArXiV Preprint](https://arxiv.org/abs/2401.03298)

If you find our work useful, kindly cite accordingly:
```
@InProceedings{benz2024enstrect,
    author       = {Benz, Christian and Rodehorst, Volker},
    title        = {Enstrect: A Stage-based Approach to 2.5D Structural Damage Detection},
    booktitle    = {TODO: European Conference on Computer Vision--Workshops},
    year         = {2024},
    organization = {Springer}
}
```

## Setup
### General
```
git clone https://github.com/ben-z-original/enstrect.git
conda create --name enstrect python=3.10
conda activate enstrect
pip install -e .
```

### PyTorch3D
Note that the PyTorch3D installation is known to be tricky. The configuration that worked in the project is:
- Ubuntu 20.04
- CUDA 12.1 with NVIDIA GeForce RTX 2080 Ti
- PyTorch 2.4.0
- Pytorch3D 0.7.7

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

## Data
The four segments (two for Bridge B and two for Bridge G) are provided in the assets. 
Note that they are part of a bigger dataset project, which will probably/hopefully released in November this year.

