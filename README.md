# ENSTRECT: A Pipeline for Enhanced Structural Inspection

ENSTRECT – short for **<ins>En</ins>hanced <ins>Str</ins>uctural Insp<ins>ect</ins>ion** – represents a workflow for 2.5D instance segmentation of **structural damages** (crack, spalling, corrosion, and exposed rebar) through *image-level segmentation/detection*, *prediction mapping* from 2D to 3D, and *damage extraction* (centerline and bounding polygon).

![Enstrect](https://github.com/user-attachments/assets/94b79295-d3c4-4101-9441-f69dbb8a6ec2)

#### Example Outputs
| Bridge B, Test Segment | Bridge G, Test Segment |
|-|-|
| <img src="https://github.com/user-attachments/assets/cbe78604-07f1-433d-944d-72679e110816" width=49%> | ![results](https://github.com/user-attachments/assets/cbe78604-07f1-433d-944d-72679e110816) |


## Citation
[ArXiV Preprint](https://arxiv.org/abs/2401.03298)

If you find our work useful, kindly cite accordingly:
```
@InProceedings{benz2024enstrect,
    author       = {Christian Benz and Volker Rodehorst},
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
Publicly available data for 2.5D or 3D structural damage detection is scarce to non-existent. Thus, we were glad to have four segments from real structures (two for *Bridge B* and two for *Bridge G*) for experimentations. Note that they are supposed to be part of a bigger dataset project, which hopefully will be released some day soon.

#### Download
The segments can be downloaded:
- by running ```python -m enstrect.datasets.download``` (which places the segments correctly in the repo tree) 
- or from [Google Drive](https://drive.google.com/file/d/1QkyoZ1o9uKuxpLIlSZ-iA9jcba46oIwW/view?usp=sharing) (correct placement in repo tree required, see below).

#### Mesh and Point Sampling
Since points clouds for higher object resolutions – as required for cracks – become overly large, only the textured meshes are provided with the above download. The examples below automatically sample points from these meshes, which are then processed by ENSTRECT. If you need high-resolution point clouds corresponding to the best image resolution, run:
- ```python -m enstrect.datasets.utils.sample_points``` for the Bridge B, test segment.
- ```python -m enstrect.datasets.utils.sample_points --help``` for information about the right command line arguments.

#### Custom Data
To apply ENSTRECT to your own data, you will need to make sure that your mesh (or point cloud) lives in a reasonable metric space (the unit is meters). If not the manually set parameters for clustering and contraction will necessarily fail. 

Furthermore, you will need to convert your camera information (both intrinsic and extrinsic parameters) into the ```cameras.json``` format. Since there isn't a universal standard for camera representation (something the computer vision community could address), this format is custom but designed to be as intuitive as possible. It directly supports camera parameters that are compatible with PyTorch3D.

If you're using camera data from Metashape, the XML file must be converted into the required JSON format. You can find the conversion script here [TODO]. For users of COLMAP, this helpful repository [Link] could provide support for creating a custom converter.

## Segmentation Model
Three models for structural damage/crack segmentation were investigated in this work. 
The first one is shipped with this repo, the others can be found in the respective repos:
- **nnU-Net-S2DS**: provided in this repo [[repo's internal path](./src/enstrect/segmentation/nnunet_s2ds.py)]
- **TopoCrack**: refer to repo [[Link](https://github.com/eesd-epfl/topo_crack_detection)]
- **DetectionHMA**: refer to repo [[Link](https://github.com/ben-z-original/detectionhma)]

The *nnU-Net-S2DS* segmentation model is supposed to be downloaded automatically when running the examples. Otherwise it can be found here [Google Drive](https://drive.google.com/file/d/1UeXzpH76GYtZtyn2IjhDvD5Qu3u91YcC/view?usp=sharing) and needs to be placed under ```src/enstrect/segmentation/checkpoints```.

The image-level output of running *nnU-Net-S2DS* on the provided ```example_image.png``` look as follows:
```bash
python -m enstrect.segmentation.nnunet_s2ds
```
![example_result](https://github.com/user-attachments/assets/b3c5215e-62c0-4ceb-a2d5-c2a3146b4eae)


Any other segmentation model can be used with ENSTRECT after being correctly wrapped into the ```SegmenterInterface```, 
see here [[repo's internal path](./src/enstrect/segmentation/base.py)].

Note that in terms of **crack segmentation** distinctly better models are now available, such as the model trained on OmniCrack30k [[Link](https://github.com/ben-z-original/omnicrack30k)].

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



# References

```
@inproceedings{benz2022image,
   author =    {Christian Benz and Volker Rodehorst},
   title =     {Image-Based Detection of Structural Defects Using Hierarchical Multi-scale Attention},
   booktitle = {Pattern Recognition. DAGM GCPR 2022. Lecture Notes in Computer Science, vol 13485},
   editor =    {Björn Andres and Florian Bernard and Daniel Cremers and Simone Frintrop and Bastian Goldlücke and Ivo Ihrke},
   isbn =      {978-3-031-16788-1},
   pages =     {337-353},
   publisher = {Springer},
   city =      {Cham},
   year =      {2022},
   doi =       {10.1007/978-3-031-16788-1_21},
}
```
```
@inproceedings{benz2024omnicrack30k,
   author =    {Christian Benz and Volker Rodehorst},
   title =     {OmniCrack30k: A Benchmark for Crack Segmentation and the Reasonable Effectiveness of Transfer Learning},
   booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
   pages =     {3876-3886},
   year =      {2024},
   url =       {https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.html},
}
```
```
@article{pantoja2022topo,
   author =  {B.G. Pantoja-Rosero and D. Oner and M. Kozinski and R. Achanta and P. Fua and F. Perez-Cruz and K. Beyer},
   title =   {TOPO-Loss for continuity-preserving crack detection using deep learning},
   journal = {Construction and Building Materials},
   issn =    {09500618},
   pages =   {128264},
   volume =  {344},
   year =    {2022},
   doi =     {10.1016/j.conbuildmat.2022.128264},
}
```


