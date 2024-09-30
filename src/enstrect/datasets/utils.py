import torch
import gdown
import zipfile
from pathlib import Path
from argparse import ArgumentParser
from pytorch3d.io import load_objs_as_meshes
from enstrect.datasets.multiview import MultiviewDataset
from utils3d.conversion.mesh import sample_points_and_save
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from utils3d.analyses.resolution import compute_resolution, compute_required_points


def download():
    parser = ArgumentParser(description="""Downloads the dataset to a given directoy path.""")
    parser.add_argument('-p', '--segments_path', type=Path,
                        default=Path(__file__).parents[1] / "assets",
                        help="Directory where the segments will be downloaded and unzipped.")
    args = parser.parse_args()

    # doẃnload and unzip plan
    zippath = Path(args.segments_path / "segments.zip")
    if not zippath.exists():
        url = "https://drive.google.com/uc?id=1QkyoZ1o9uKuxpLIlSZ-iA9jcba46oIwW"
        gdown.download(url, str(zippath), quiet=False)
        with zipfile.ZipFile(str(zippath), 'r') as zip_ref:
            zip_ref.extractall(str(args.segments_path / "segments"))
    else:
        raise RuntimeError(f"The segments dataset already exists in the path: {zippath}")


def sample_points_based_on_resolution(mesh_path, cameras_path, out_ply, device="cuda:0"):
    meshes_pyt3d = load_objs_as_meshes([mesh_path]).to(device)
    dataset = MultiviewDataset(cameras_path)
    camera, min_depth = compute_minimal_distance(meshes_pyt3d, dataset)

    # compute number of required points
    resolution_mmpx, resolution_mm2_px = compute_resolution(camera, min_depth)
    num_points = compute_required_points(meshes_pyt3d, resolution_mm2_px)

    sample_points_and_save(meshes_pyt3d, out_ply, num_points=num_points)


def compute_minimal_distance(meshes_pyt3d, dataset):
    rasterizer = MeshRasterizer(raster_settings=RasterizationSettings(blur_radius=0.0, faces_per_pixel=1))

    # loop over views and compute depth/distance
    cameras, min_depths = [], []
    for sample in dataset:
        cam = sample["camera"]
        rasterizer.raster_settings.image_size = cam.image_size.cpu().to(torch.int64).tolist()[0]
        fragments = rasterizer(meshes_pyt3d, cameras=cam)
        depth = fragments.zbuf[0, ..., 0]

        cameras.append(cam)
        min_depths.append(depth[depth != -1].min())

    min_depths = torch.tensor(min_depths)
    idx_min = min_depths.argmin().item()

    return cameras[idx_min], min_depths[idx_min]


def sample_points_from_meshes_pyt3d(meshes, num_points=1000000):
    torch.manual_seed(42)
    points, normals, colors = sample_points_from_meshes(meshes,
                                                        return_normals=True, return_textures=True,
                                                        num_samples=num_points)
    pcd_pyt3d = Pointclouds(points, normals, colors)
    return pcd_pyt3d


def sample_points():
    parser = ArgumentParser(description="""Sample points from mesh according to the best image resolution.""")
    parser.add_argument('-m', '--mesh_path', type=Path,
                        default=Path(__file__).parents[1] / "assets" / "segments" / "bridge_b" /
                                "segment_test" / "mesh" / "mesh.obj",
                        help="Path to the file that contains the intrinsic and extrinsic camera information.")
    parser.add_argument('-c', '--cameras_path', type=Path,
                        default=Path(__file__).parents[1] / "assets" / "segments" / "bridge_b" /
                                "segment_test" / "cameras.json",
                        help="Path to the file that contains the intrinsic and extrinsic camera information.")
    parser.add_argument('-o', '--out_ply', type=Path,
                        default=Path(__file__).parents[1] / "assets" / "segments" / "bridge_b" /
                                "segment_test" / "pcd.ply",
                        help="Path to the file in ply format where the resulting point cloud will be stored.")
    args = parser.parse_args()

    sample_points_based_on_resolution(args.mesh_path, args.cameras_path, args.out_ply)
