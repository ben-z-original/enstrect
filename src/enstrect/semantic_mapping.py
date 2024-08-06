import cv2
import torch
import matplotlib
import numpy as np
import open3d as o3d
from addict import Dict
from pathlib import Path
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
from omnicrack30k.inference import OmniCrack30kModel
from utils3d.parser.parser import parse_agisoft_xml
from utils3d.conversion.cameras import views2pytorch3d_cameras
from utils3d.datasets.multiview_dataset import MapperDataset
from torch.utils.data import Dataset, DataLoader
from utils3d.conversion.cloud import pynt_to_pyt3d, pynt_to_o3d

matplotlib.use("TkAgg")


class SemanticMapper:
    def __init__(self, model, fuser=None, device="cuda:0"):
        self.model = model
        self.device = device

    def __call__(self, pcd_pynt, dataset):
        pcd_pyt3d = pynt_to_pyt3d(pcd_pynt, self.device)
        # point_cloud_pyt3d = point_cloud_pyt3d.to(self.device)

        features = Dict({feat: np.zeros((len(pcd_pyt3d.points_packed()), len(dataset)), dtype=np.float16)
                         for feat in ["crack", "distances", "angles", "visible"]})

        for i, sample in enumerate(dataset):
            soft = self.model.pyt_inference(sample["image"])[1][1]

            # get image coordinates
            coords = self.get_image_coordinates(pcd_pyt3d, sample["camera"])

            # get features
            features.distances[:, i] = self.compute_distances(pcd_pyt3d, sample["camera"])
            features.angles[:, i] = self.compute_angular_view_deviation(pcd_pyt3d, sample["camera"])
            features.visible[:, i] = self.compute_visibility(pcd_pyt3d, sample["camera"], coords)

            coords[features.visible[:, i] != 1, :] *= 0

            # probabilities
            features.crack[:, i] = soft[coords[:, 1], coords[:, 0]]
            features.crack[features.visible[:, i] != 1, i] = np.nan

        aggr = np.nanmean(features.crack, axis=1)
        aggr = np.nan_to_num(aggr, nan=0.0)

        pcd_pynt.points["crack"] = np.float32(aggr)
        pcd_pynt.points["defect"] = np.ubyte(0.5 < aggr)
        #np.zeros((len(ply_pred.points),), np.ubyte)
        return pcd_pynt

    @staticmethod
    def get_image_coordinates(point_cloud_pyt3d, camera):
        coords = camera.transform_points_screen(point_cloud_pyt3d.points_packed())
        coords = coords.to(torch.int)[:, :2].cpu().numpy()
        return coords

    @staticmethod
    def compute_distances(point_cloud_pyt3d, camera):
        distances = torch.linalg.norm(
            point_cloud_pyt3d.points_packed() - camera.get_camera_center(),
            dim=1).cpu().numpy()
        return distances

    @staticmethod
    def compute_angular_view_deviation(point_cloud_pyt3d, camera):
        cam_view_direction = (camera.R @ torch.tensor([0, 0, 1.0], device=camera.device))
        cam_view_direction /= torch.linalg.norm(cam_view_direction)
        angles = torch.rad2deg(torch.arccos(point_cloud_pyt3d.normals_packed() @ cam_view_direction.T)).squeeze()
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


def run(ply_path, cameras_xml_path, images_dir):
    pcd_pynt = PyntCloud.from_file(str(ply_path))
    model = OmniCrack30kModel()
    dataset = MapperDataset(cameras_xml_path, images_dir)

    mapper = SemanticMapper(model)
    pcd_pynt = mapper(pcd_pynt, dataset)
    return pcd_pynt


if __name__ == "__main__":
    ply_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud.ply"
    images_dir = "/media/chrisbe/backup/segments/bridge_b/segment1/real/original"
    cameras_xml_path = "/media/chrisbe/backup/segments/bridge_b/segment1/real/cameras.xml"
    ply_out_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud_tmp.ply"

    pcd_pynt = run(ply_path, cameras_xml_path, images_dir)
    pcd_pynt.to_file(ply_out_path)

    pcd_o3d = pynt_to_o3d(pcd_pynt, "crack")
    o3d.visualization.draw_geometries([pcd_o3d])
