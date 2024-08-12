import torch
import matplotlib
import numpy as np
import open3d as o3d
from addict import Dict
from utils3d.fusion.static import MeanFuser
from utils3d.conversion.cloud import pynt_to_pyt3d

matplotlib.use("TkAgg")


class SemanticMapper:
    def __init__(self, model, fuser=MeanFuser(), device="cuda:0"):
        self.model = model
        self.device = device
        self.fuser = fuser
        self.classes = ["background", "crack"]

    def __call__(self, pcd_pynt, dataset):
        pcd_pyt3d = pynt_to_pyt3d(pcd_pynt, self.device)

        viewing_conditions = Dict({feat: np.zeros((len(pcd_pyt3d.points_packed()), len(dataset)), dtype=np.float16)
                                   for feat in ["distances", "angles", "visible"]})

        probabilities = torch.zeros((len(pcd_pyt3d.points_packed()), len(dataset), len(self.classes)),
                                    dtype=torch.float16)

        for i, sample in enumerate(dataset):
            # get image coordinates
            coords = self.get_image_coordinates(pcd_pyt3d, sample["camera"])

            # get viewing conditions
            viewing_conditions.distances[:, i] = self.compute_distances(pcd_pyt3d, sample["camera"])
            viewing_conditions.angles[:, i] = self.compute_angular_view_deviation(pcd_pyt3d, sample["camera"])
            viewing_conditions.visible[:, i] = self.compute_visibility(pcd_pyt3d, sample["camera"], coords)

            coords[viewing_conditions.visible[:, i] != 1, :] *= 0

            # probabilities
            soft = self.model.pyt_inference(sample["image"])[1]
            probabilities[:, i] = soft[:, coords[:, 1], coords[:, 0]].T
            probabilities[viewing_conditions.visible[:, i] != 1, i] = np.nan

        aggr, argmax = self.fuser(probabilities)

        for i, c in enumerate(self.classes):
            pcd_pynt.points[c] = np.float32(aggr[:, i].cpu().numpy())

        pcd_pynt.points["argmax"] = np.ubyte(argmax.cpu().numpy())
        return pcd_pynt

    @staticmethod
    def get_image_coordinates(pcd_pyt3d, camera):
        coords = camera.transform_points_screen(pcd_pyt3d.points_packed())
        coords = coords.to(torch.int)[:, :2].cpu().numpy()
        return coords

    @staticmethod
    def compute_distances(pcd_pyt3d, camera):
        distances = torch.linalg.norm(
            pcd_pyt3d.points_packed() - camera.get_camera_center(),
            dim=1).cpu().numpy()
        return distances

    @staticmethod
    def compute_angular_view_deviation(pcd_pyt3d, camera):
        cam_view_direction = (camera.R @ torch.tensor([0, 0, 1.0], device=camera.device))
        cam_view_direction /= torch.linalg.norm(cam_view_direction)
        angles = torch.rad2deg(torch.arccos(pcd_pyt3d.normals_packed() @ cam_view_direction.T)).squeeze()
        angles = angles.cpu().numpy()
        return angles

    @staticmethod
    def compute_visibility(pcd_pyt3d, camera, coords, self_occlusion=True):
        visibility = ((0 <= coords[:, 0]) * (coords[:, 0] < int(camera.image_size[0][1].item())) *
                      (0 <= coords[:, 1]) * (coords[:, 1] < int(camera.image_size[0][0].item())))

        if self_occlusion:
            # TODO: test
            pcd_o3d = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.float64(pcd_pyt3d.points_packed().cpu().numpy()).copy())
            )
            # pcd_o3d = pcd_o3d.select_by_index(np.where(visibility)[0]) # TODO: idxs will change
            diameter = np.linalg.norm(np.asarray(pcd_o3d.get_max_bound()) - np.asarray(pcd_o3d.get_min_bound()))
            _, idxs = pcd_o3d.hidden_point_removal(
                camera.get_camera_center()[0].cpu().to(torch.float64).numpy()[:, None].copy(),
                100 * diameter)

            visibility[idxs] = 0

        return visibility
