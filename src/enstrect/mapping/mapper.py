import torch
import matplotlib
import numpy as np
import open3d as o3d
from addict import Dict
from enstrect.mapping.fuser import Fuser
from enstrect.extraction.utils import pynt_to_pyt3d

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt


class Mapper:
    def __init__(self, model, fuser=Fuser(), device="cuda:0"):
        self.model = model
        self.device = device
        self.fuser = fuser

    def __call__(self, pcd_pynt, dataset, store_probabilities=True):
        pcd_pyt3d = pynt_to_pyt3d(pcd_pynt, self.device)

        viewing_conditions = Dict({feat: np.zeros((len(pcd_pyt3d.points_packed()), len(dataset)), dtype=np.float16)
                                   for feat in ["distances", "angles", "visible"]})

        probabilities = torch.zeros((len(pcd_pyt3d.points_packed()), len(dataset), len(self.model.classes)),
                                    dtype=torch.float16)

        for i, sample in enumerate(dataset):
            # get image coordinates
            coords = self.get_image_coordinates(pcd_pyt3d, sample["camera"])

            # get viewing conditions
            viewing_conditions.distances[:, i] = self.compute_distances(pcd_pyt3d, sample["camera"])
            viewing_conditions.angles[:, i] = self.compute_angular_view_deviation(pcd_pyt3d.normals_packed(),
                                                                                  sample["camera"])
            viewing_conditions.visible[:, i] = self.compute_visibility(pcd_pyt3d, sample["camera"], coords, True)

            coords[viewing_conditions.visible[:, i] != 1, :] *= 0

            # probabilities
            soft = self.model(sample["image"])[0]

            if False:
                matplotlib.use("TkAgg")
                plt.subplot(121)
                plt.imshow(sample["image"].moveaxis(0, -1)[..., [2, 1, 0]].to(torch.uint8))
                plt.subplot(122)
                soft[1, 0, 0] = 0
                soft[1, -1, -1] = 1
                plt.imshow(soft[1], 'gray')
                plt.show()

            probabilities[:, i] = soft[:, coords[:, 1], coords[:, 0]].T
            probabilities[viewing_conditions.visible[:, i] != 1, i] = np.nan

        aggr, argmax = self.fuser(probabilities, viewing_conditions)

        pcd_pynt.points["argmax"] = np.ubyte(argmax.cpu().numpy())

        for i, cl in enumerate(self.model.classes):
            pcd_pynt.points[cl] = np.ubyte(np.array(pcd_pynt.points["argmax"]) == i)
            if store_probabilities:
                pcd_pynt.points[cl + "_probabilities"] = np.float32(aggr[:, i].cpu().numpy())

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
    def compute_angular_view_deviation(normals, camera):
        cam_view_direction = (camera.R @ torch.tensor([0, 0, 1.0], device=camera.device))
        cam_view_direction /= torch.linalg.norm(cam_view_direction)
        angles = torch.rad2deg(torch.arccos(normals @ cam_view_direction.T)).squeeze()
        angles = angles.cpu().numpy()
        return angles

    @staticmethod
    def compute_visibility(pcd_pyt3d, camera, coords, self_occlusion=True):
        visibility = ((0 <= coords[:, 0]) * (coords[:, 0] < int(camera.image_size[0][1].item())) *
                      (0 <= coords[:, 1]) * (coords[:, 1] < int(camera.image_size[0][0].item())))

        if self_occlusion:
            # prepare point cloud (hidden point removal algo implemented in open3d)
            points = np.float64(pcd_pyt3d.points_packed().cpu().numpy()).copy()
            normals = np.float64(pcd_pyt3d.normals_packed().cpu().numpy()).copy()
            pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pcd_o3d.normals = o3d.utility.Vector3dVector(normals)

            # run hidden point removal (HPR)
            idxs_visible = np.where(visibility)[0]
            pcd_o3d = pcd_o3d.select_by_index(idxs_visible)
            diameter = np.linalg.norm(np.asarray(pcd_o3d.get_max_bound()) - np.asarray(pcd_o3d.get_min_bound()))
            _, idxs = pcd_o3d.hidden_point_removal(
                camera.get_camera_center()[0].cpu().to(torch.float64).numpy()[:, None].copy(),
                1000 * diameter)#10 * diameter)  # 1000
            idxs_remove = list(set(range(len(idxs_visible))).difference(idxs))

            visibility[idxs_visible[idxs_remove]] = 0

            if False:
                pcd_o3d = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(np.float64(pcd_pyt3d.points_packed().cpu().numpy()).copy())
                )
                pcd_o3d.normals = o3d.utility.Vector3dVector(
                    np.float64(pcd_pyt3d.normals_packed().cpu().numpy()).copy())
                pcd_o3d.colors = o3d.utility.Vector3dVector(
                    np.float64(np.vstack([visibility, visibility, visibility]).T))

                o3d.visualization.draw_geometries([pcd_o3d])

        return visibility
