import cv2
import torch
import matplotlib
import numpy as np
import open3d as o3d
from addict import Dict
from pathlib import Path
from pytorch3d.io import IO
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
from omnicrack30k.inference import OmniCrack30kModel
from utils3d.parser.parser import parse_agisoft_xml
from utils3d.conversion.cameras import views2pytorch3d_cameras
from utils3d.datasets.multiview_dataset import MapperDataset

matplotlib.use("TkAgg")


class SemanticMapper:
    def __init__(self, model, fuser=None, device="cuda:0"):
        self.model = model
        self.device = device

    def __call__(self, point_cloud_pyt3d, dataset):
        point_cloud_pyt3d = point_cloud_pyt3d.to(self.device)

        features = Dict({feat: np.zeros((len(point_cloud_pyt3d.points_packed()), len(dataset)), dtype=np.float16)
                         for feat in ["crack", "distances", "angles", "visible"]})

        for i, (name, camera, image) in enumerate(dataset):
            soft = self.model.pyt_inference(image)[1][0]

            # get image coordinates
            coords = self.get_image_coordinates(point_cloud_pyt3d, camera)

            # get features
            features.distances[:, i] = self.compute_distances(point_cloud_pyt3d, camera)
            features.angles[:, i] = self.compute_angular_view_deviation(point_cloud_pyt3d, camera)
            features.visible[:, i] = self.compute_visibility(cloud_o3d, coords, camera)

            coords[features.visible[:, i] != 1, :] *= 0

            # probabilities
            features.crack[:, i] = soft[coords[:, 1], coords[:, 0]]
            features.crack[features.visible[:, i] != 1, i] = np.nan

        aggr = np.nanmean(features.crack, axis=1)
        aggr = np.nan_to_num(aggr, nan=1.0)

        colors = np.repeat(aggr[:, None], 3, axis=1)

        return colors

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
    def compute_visibility(cloud_o3d, coords, camera):
        # features.visible[:, i] = ((0 <= coords[:, 0]) * (coords[:, 0] < int(camera.image_size[0][1].item())) *
        #                          (0 <= coords[:, 1]) * (coords[:, 1] < int(camera.image_size[0][0].item())))
        visibility = ((0 <= coords[:, 0]) * (coords[:, 0] < int(camera.image_size[0][1].item())) *
                      (0 <= coords[:, 1]) * (coords[:, 1] < int(camera.image_size[0][0].item())))
        #coords[:, 0] = np.clip(coords[:, 0], 0, WIDTH - 1)
        #coords[:, 1] = np.clip(coords[:, 1], 0, HEIGHT - 1)

        if False:
            cloud_tmp = cloud_o3d.select_by_index(np.where(features.visible[:, i])[0])
            _, idxs = cloud_tmp.hidden_point_removal(
                camera.get_camera_center()[0].cpu().to(torch.float64).numpy()[:, None],
                100 * float(features.distances[:, i].max()))

            features.visible[idxs, i] = 0

        return visibility


obj_filename = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud.ply"
point_cloud_pyt3d = IO(True).load_pointcloud(path=obj_filename)
cloud_o3d = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(np.float64(point_cloud_pyt3d.points_packed().cpu().numpy())))
cloud_pynt = PyntCloud.from_file(str(obj_filename))

model = OmniCrack30kModel()
views_dir = "/media/chrisbe/backup/segments/bridge_b/segment1/real/original"
cameras_xml_path = "/media/chrisbe/backup/segments/bridge_b/segment1/real/cameras.xml"

dataset = MapperDataset(cameras_xml_path, views_dir)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

mapper = SemanticMapper(model)

colors = mapper(point_cloud_pyt3d, dataset)

# cloud_o3d = cloud_o3d.select_by_index(np.where(features.visible[:, i])[0])
cloud_o3d.colors = o3d.utility.Vector3dVector(np.float64(colors))
cloud_o3d.normals = o3d.utility.Vector3dVector(np.float64(point_cloud_pyt3d.normals_packed().cpu().numpy()))
o3d.visualization.draw_geometries(
    [cloud_o3d])  # , show_skybox=False)# bg_color=(0.1, 0.1, 0.1, 1.0), show_skybox=False)

cloud_pynt = PyntCloud.from_file(str(obj_filename))
cloud_pynt.points["crack"] = np.float32(colors[:, 0])
cloud_pynt.to_file("/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud_tmp.ply")
print()

views = parse_agisoft_xml(cameras_xml_path)
# views = {view: views[view] for i, view in enumerate(views.keys()) if i % 8 == 0}
views = {view: views[view] for i, view in enumerate(views.keys()) if 1 < i and i < 3}
scale = 1 / 4
cameras = views2pytorch3d_cameras(views, device=device, scale=scale)

HEIGHT = int(cameras[0].image_size[0][0].item())
WIDTH = int(cameras[0].image_size[0][1].item())

# probs = torch.zeros((len(point_cloud.points_packed()), len(cameras)), dtype=torch.half)
probs = np.zeros((len(point_cloud_pyt3d.points_packed()), len(cameras)), dtype=np.float16)

features = Dict({feat: np.zeros((len(point_cloud_pyt3d.points_packed()), len(cameras)), dtype=np.float16)
                 for feat in ["crack", "distances", "angles", "visible"]})

for i, (view, cam) in enumerate(zip(views, cameras)):
    img = cv2.imread(f"/media/chrisbe/backup/segments/bridge_b/segment1/real/original/{view}.JPG", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    soft = model(img)[0]

    # get image coordinates
    coords = cam.transform_points_screen(point_cloud_pyt3d.points_packed())
    coords = coords.to(torch.int)[:, :2].cpu().numpy()

    # distances
    features.distances[:, i] = torch.linalg.norm(point_cloud_pyt3d.points_packed() - cam.get_camera_center(),
                                                 dim=1).cpu().numpy()

    # compute angular view deviation
    cam_view_direction = (cam.R @ torch.tensor([0, 0, 1.0], device=device))
    cam_view_direction /= torch.linalg.norm(cam_view_direction)
    angles = torch.rad2deg(torch.arccos(point_cloud_pyt3d.normals_packed() @ cam_view_direction.T)).squeeze()
    features.angles[:, i] = angles.cpu().numpy()

    # visibility
    features.visible[:, i] = ((0 <= coords[:, 0]) * (coords[:, 0] < WIDTH) *
                              (0 <= coords[:, 1]) * (coords[:, 1] < HEIGHT))
    coords[features.visible[:, i] != 1, :] *= 0
    cloud_tmp = cloud_o3d.select_by_index(np.where(features.visible[:, i])[0])
    _, idxs = cloud_tmp.hidden_point_removal(cam.get_camera_center()[0].cpu().to(torch.float64).numpy()[:, None],
                                             100 * float(features.distances[:, i].max()))

    features.visible[idxs, i] = 0

    # probabilites
    features.crack[:, i] = soft[coords[:, 1], coords[:, 0]]
    features.crack[features.visible[:, i] != 1, i] = np.nan

    # cloud_tmp = cloud_tmp.select_by_index(idxs)
    # o3d.visualization.draw([cloud_tmp])

# to lines
# analysis

aggr = np.nanmean(features.crack, axis=1)
aggr = np.nan_to_num(aggr, nan=1.0)

colors = np.repeat(aggr[:, None], 3, axis=1)

# cloud_o3d = cloud_o3d.select_by_index(np.where(features.visible[:, i])[0])
cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
cloud_o3d.normals = o3d.utility.Vector3dVector(point_cloud_pyt3d.normals_packed().cpu().numpy())
o3d.visualization.draw([cloud_o3d], bg_color=(0.1, 0.1, 0.1, 1.0), show_skybox=False)
