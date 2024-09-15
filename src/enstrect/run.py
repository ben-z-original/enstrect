import warnings
import open3d as o3d
from pathlib import Path
from pytorch3d.io import IO
from pyntcloud import PyntCloud
from argparse import ArgumentParser
from enstrect.mapping.fuser import Fuser, AngleBestFuser, NaiveMaxFuser, NaiveMeanFuser, AngleRangeFuser
from enstrect.mapping.mapper import Mapper
from utils3d.conversion.lines import G_to_obj
from utils3d.conversion.cloud import pyt3d_to_pynt
from enstrect.datasets.multiview import MultiviewDataset
from enstrect.extraction.utils import prepare_exposed_rebar
from enstrect.segmentation.nnunet_s2ds import NNUNetS2DSModel
from enstrect.segmentation.omnicrack30k_model import OmniCrack30kModel
from enstrect.extraction.contraction import extract_centerlines
from enstrect.extraction.boundary import extract_bounding_polygons
from utils3d.visualization.orbiting_gif import mesh_o3d_to_gif
from enstrect.datasets.utils import sample_points_from_meshes_pyt3d
from utils3d.visualization.visualize_o3d import plot_o3d_mesh_with_lines

warnings.filterwarnings("ignore", message="Detected old nnU-Net plans")  # warning can be ignored


def run(obj_or_ply_path, cameras_path, images_dir, out_dir, select_views, num_points, scale,
        fuser=AngleRangeFuser, model=OmniCrack30kModel):
    # create outdir
    out_dir.mkdir(exist_ok=True)

    # sample or load point cloud
    if obj_or_ply_path.suffix == ".obj":
        meshes = IO().load_mesh(obj_or_ply_path)
        pcd_pyt3d = sample_points_from_meshes_pyt3d(meshes, num_points=num_points)
        pcd_pynt = pyt3d_to_pynt(pcd_pyt3d)
    else:
        pcd_pynt = PyntCloud.from_file(str(obj_or_ply_path))

    # set up segmentation model, mapper, and dataset
    #model = model()
    model = NNUNetS2DSModel()
    fuser = fuser(class_weight=model.class_weight)
    mapper = Mapper(model, fuser=fuser)
    dataset = MultiviewDataset(cameras_path, images_dir, select_views=select_views, scale=scale)

    if True:
        # perform mapping 2D->3D
        pcd_pynt = mapper(pcd_pynt, dataset, out_dir)
        pcd_pynt = prepare_exposed_rebar(pcd_pynt)
        out_dir.mkdir(exist_ok=True)
        pcd_pynt.to_file(str(Path(out_dir) / f"pcd_{num_points}_processed.ply"))
    else:
        pcd_pynt = PyntCloud.from_file(str(Path(out_dir) / f"pcd_{num_points}_processed.ply"))

    # perform extraction
    G_dict = {
        "crack": extract_centerlines(pcd_pynt, category="crack"),
        "spalling": extract_bounding_polygons(pcd_pynt, category="spalling"),
        "corrosion": extract_bounding_polygons(pcd_pynt, category="corrosion"),
        "exposed_rebar": extract_bounding_polygons(pcd_pynt, category="exposed_rebar")
    }

    # export centerlines and bounding polygons as obj
    for key, G in G_dict.items():
        G_to_obj(G, str(Path(out_dir) / f"{key}.obj"))

    # visualize centerlines and bounding polygons on top of mesh
    mesh_o3d = o3d.io.read_triangle_mesh(str(obj_or_ply_path), True)

    mesh_o3d_to_gif(mesh_o3d,
                    G_dict=G_dict,
                    gif_path=str(Path(out_dir) / f"results.gif"),
                    off_screen=True,
                    full_orbit=True if "crackensembles" in str(obj_or_ply_path) else False)
    # plot_o3d_mesh_with_lines(mesh_o3d, G_dict)


if __name__ == "__main__":
    parser = ArgumentParser(description="""Run enstrect.""")
    parser.add_argument('-p', '--obj_or_ply_path', type=Path,
                        default=Path(__file__).parent / "assets" / "segments" / "bridge_g" /
                                "segment_dev" / "mesh" / "mesh.obj",
                        help="Path to the mesh (.obj; points will be sampled) or point cloud file (.ply).")
    parser.add_argument('-i', '--images_dir', type=Path,
                        default=Path(__file__).parent / "assets" / "segments" / "bridge_g" /
                                "segment_dev" / "views",
                        help="Path to the directory which contains the images/views.")
    parser.add_argument('-c', '--cameras_path', type=Path,
                        default=Path(__file__).parent / "assets" / "segments" / "bridge_g" /
                                "segment_dev" / "cameras.json",
                        help="Path to the file that contains the intrinsic and extrinsic camera information.")
    parser.add_argument('-o', '--out_dir', type=Path,
                        default=Path(__file__).parent / "assets" / "segments" / "bridge_g" /
                                "segment_dev" / "out",
                        help="Path to the directory where the results will be stored")
    parser.add_argument('-v', '--select_views', nargs='*', #default=["0005", "0006"],
                        help="")
    parser.add_argument('-s', '--scale', type=float, default=1.0, #1.0,  # 1.0,  # 0.25,
                        help="Rescale the images to be processed. Defaults is 0.25 to reduce runtime.")
    parser.add_argument('-n', '--num_points', type=int, default=1000000,
                        help="Number of points to sample from the mesh. Default is 1,000,000")
    args = parser.parse_args()

    run(args.obj_or_ply_path, args.cameras_path, args.images_dir,
        args.out_dir, args.select_views, args.num_points, args.scale)
