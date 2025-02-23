# ENSTRECT: A Pipeline for Enhancing Structural Inspection

ENSTRECT – short for **<ins>En</ins>hanced <ins>Str</ins>uctural Insp<ins>ect</ins>ion** – represents a workflow for 2.5D instance segmentation of **structural damages** (crack, spalling, corrosion, and exposed rebar) through *image-level segmentation/detection*, *prediction mapping* from 2D to 3D, and *damage extraction* (centerline and bounding polygon).

![Enstrect](https://github.com/user-attachments/assets/94b79295-d3c4-4101-9441-f69dbb8a6ec2)

#### Example Outputs
| Bridge B, Test Segment | Bridge G, Test Segment |
|-|-|
| <img src="https://github.com/user-attachments/assets/cc4ae4b5-46a7-4f21-9ba5-288f338c197d" width=100%> | <img src="https://github.com/user-attachments/assets/5a6c6b74-d500-4e9e-9cb3-deaf048c6d2f" width=100%> |

## Citation
[Link to ECCVW'24 Paper](https://arxiv.org/pdf/2401.03298) (ArXiV)

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
``` bash
# create and activate conda environment
conda create --name enstrect python=3.10
conda activate enstrect

# clone and install repository
git clone https://github.com/ben-z-original/enstrect.git
cd enstrect
pip install -e .
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7  # needs knowledge about installed torch version
```

#### PyTorch3D Issues
PyTorch3D installation can be tricky. This especially refers to the alignment of versions of CUDA, PyTorch, and PyTorch3D. The configuration that was working in the project is:
- Ubuntu 20.04
- CUDA 12.1 with NVIDIA GeForce RTX 2080 Ti
- PyTorch 2.4.1
- Pytorch3D 0.7.7

## Data
Publicly available data for 2.5D or 3D structural damage detection is scarce to non-existent. Thus, we were glad to have four segments from real structures (two for *Bridge B* and two for *Bridge G*) for experimentations.

#### Download
The segments can be downloaded:
- by running ```python -m enstrect.datasets.download``` (which places the segments correctly in the repo tree) 
- or from [Google Drive](https://drive.google.com/file/d/1QkyoZ1o9uKuxpLIlSZ-iA9jcba46oIwW/view?usp=sharing) (correct placement in repo tree required, see below).

#### Custom Data
To apply ENSTRECT to your own data, you will need to make sure that your mesh (or point cloud) lives in a reasonable metric space (the unit is meters). If not, extraction will fail due to unsuited parameters for clustering and contraction. 

Furthermore, you will need to convert your camera information (both intrinsic and extrinsic parameters) into the ```cameras.json``` format. Since there isn't a universal standard for camera representation, this format is custom but intuitive. It directly supports camera parameters that are compatible with PyTorch3D. If you use Metashape, COLMAP, etc. you need to transform the cameras in the PyTorch3D compatible format.

For *Bridge G (dev)* the beginning of the ```cameras.json``` looks like:
```python
{
  "0000": {
    "focal_length": [
      [
        11568.5576171875,
        11568.5576171875
      ]
    ],
    "principal_point": [
      [
        3746.271728515625,  # ~width/2 (swapped compared to image_size; PyTorch3D convention)
        2372.466064453125   # ~height/2 
      ]
    ],
    "image_size": [
      [
        4912.0,             # height
        7360.0              # width
      ]
    ],
    "R": [
      [
        [
          -0.8191599249839783,
          -0.027572205290198326,
          -0.5729021430015564
        ],
        [
          -0.013821378350257874,
          0.9995027780532837,
          -0.02834092080593109
        ],
        [
          0.5733987092971802,
          -0.01529744639992714,
          -0.8191335797309875
        ]
      ]
    ],
    "T": [
      [
        0.5004259943962097,
        0.7711246609687805,
        3.7751145362854004
      ]
    ],
    "in_ndc": false
  },
  "0001": ...
```

## Segmentation Model
Three models for structural damage/crack segmentation were investigated in this work. 
The first one is shipped with this repo, the others can be found in the respective repos:
- **nnU-Net-S2DS**: provided in this repo [[repo's internal path](./src/enstrect/segmentation/nnunet_s2ds.py)]
- **TopoCrack**: refer to repo [[Link](https://github.com/eesd-epfl/topo_crack_detection)]
- **DetectionHMA**: refer to repo [[Link](https://github.com/ben-z-original/detectionhma)]

The *nnU-Net-S2DS* segmentation model is downloaded automatically when running the examples. Otherwise the weights can be found here [Google Drive](https://drive.google.com/file/d/1UeXzpH76GYtZtyn2IjhDvD5Qu3u91YcC/view?usp=sharing) and need to be placed under ```src/enstrect/segmentation/checkpoints```.

The image-level output of running *nnU-Net-S2DS* on the provided ```example_image.png``` look as follows:
```bash
python -m enstrect.segmentation.nnunet_s2ds
```
![example_result](https://github.com/user-attachments/assets/24754c4f-41cc-4251-a8bf-90075146fe8e)


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
For running the *Bridge B (test)* example execute (the paths to the mesh are stored in the defaults):
``` bash
python -m enstrect.run
```
Note that by default the image resolution is reduced to 0.25 compared to the original scale for faster (exemplary) processing. For best quality run at 1.0 image resolution, which naturally takes longer.

The results are stored in the ```assets/segments/bridge_b/segment_test/out``` directory. They contain class-wise ```obj``` files with the extracted medial axis or boundaries and a ```ply``` point cloud with attributes such as *crack*, *spalling*, etc. Both you can visualized with, e.g., [CloudCompare](https://cloudcompare-org.danielgm.net/release/).


### Exposed Rebar
For running the *Bridge G (test)* segment execute (from the repository's root path):
```bash
python -m enstrect.run \
    --obj_or_ply_path src/enstrect/assets/segments/bridge_g/segment_test/mesh/mesh.obj \
    --images_dir src/enstrect/assets/segments/bridge_g/segment_test/views \
    --cameras_path src/enstrect/assets/segments/bridge_g/segment_test/cameras.json \
    --out_dir src/enstrect/assets/segments/bridge_g/segment_test/out \
    --scale 0.5
```
For demonstration of the ```scale``` parameter, a scale of 0.5 was set.


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


