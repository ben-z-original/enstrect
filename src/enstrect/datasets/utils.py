import json
import torch
from addict import Dict
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.ops import sample_points_from_meshes


def json_to_pyt3d(cameras_path, select_views=None, scale=1.0, device="cuda:0"):
    with open(cameras_path, 'r') as stream:
        cameras_json = Dict(json.load(stream))

    view_keys, cameras = [], []
    for key, cam in cameras_json.items():
        if select_views is not None and key not in select_views:
            continue
        camera = PerspectiveCameras(
            focal_length=((scale * cam.focal_length[0][0],
                           scale * cam.focal_length[0][1]),),
            principal_point=((scale * cam.principal_point[0][0],
                              scale * cam.principal_point[0][1]),),
            image_size=((int(scale * cam.image_size[0][0]),
                         int(scale * cam.image_size[0][1])),),
            R=cam.R, T=cam.T, in_ndc=False, device=device)
        view_keys.append(key)
        cameras.append(camera)
    return view_keys, cameras


def sample_points_from_meshes_pyt3d(meshes, num_points=1000000):
    torch.manual_seed(42)
    points, normals, colors = sample_points_from_meshes(meshes,
                                                        return_normals=True, return_textures=True,
                                                        num_samples=num_points)
    pcd_pyt3d = Pointclouds(points, normals, colors)
    return pcd_pyt3d
