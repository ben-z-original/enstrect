import warnings
from pathlib import Path
from pytorch3d.io import IO
from pyntcloud import PyntCloud
from argparse import ArgumentParser
from enstrect.mapping.fuser import Fuser
from enstrect.mapping.mapper import Mapper
from enstrect.extraction.utils import G_to_obj
from enstrect.extraction.utils import pyt3d_to_pynt
from enstrect.datasets.multiview import MultiviewDataset
from enstrect.extraction.utils import prepare_exposed_rebar
from enstrect.segmentation.nnunet_s2ds import NNUNetS2DSModel
from enstrect.extraction.contraction import extract_centerlines
from enstrect.extraction.boundary import extract_bounding_polygons
from enstrect.datasets.utils import sample_points_from_meshes_pyt3d

warnings.filterwarnings("ignore", message="Detected old nnU-Net plans")  # warning can be ignored


def run(obj_or_ply_path, cameras_path, images_dir, out_dir, select_views, num_points, scale):
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
    model = NNUNetS2DSModel()
    fuser = Fuser(class_weight=model.class_weight)
    mapper = Mapper(model, fuser=fuser)
    dataset = MultiviewDataset(cameras_path, images_dir, select_views=select_views, scale=scale)

    # perform mapping 2D->3D
    pcd_pynt = mapper(pcd_pynt, dataset)
    pcd_pynt = prepare_exposed_rebar(pcd_pynt)
    out_dir.mkdir(exist_ok=True)
    pcd_pynt.to_file(str(Path(out_dir) / f"pcd_{num_points}_processed.ply"))

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


if __name__ == "__main__":
    parser = ArgumentParser(description="""Run enstrect for segmenting, mapping, and extraction structural damage.""")
    parser.add_argument('-p', '--obj_or_ply_path', type=Path,
                        default=Path(__file__).parent / "assets" / "segments" / "bridge_b" /
                                "segment_test" / "mesh" / "mesh.obj",
                        help="Path to the mesh (.obj; points will be sampled) or point cloud file (.ply).")
    parser.add_argument('-i', '--images_dir', type=Path,
                        default=Path(__file__).parent / "assets" / "segments" / "bridge_b" /
                                "segment_test" / "views",
                        help="Path to the directory which contains the images/views.")
    parser.add_argument('-c', '--cameras_path', type=Path,
                        default=Path(__file__).parent / "assets" / "segments" / "bridge_b" /
                                "segment_test" / "cameras.json",
                        help="Path to the file that contains the intrinsic and extrinsic camera information.")
    parser.add_argument('-o', '--out_dir', type=Path,
                        default=Path(__file__).parent / "assets" / "segments" / "bridge_b" /
                                "segment_test" / "out",
                        help="Path to the directory where the results will be stored")
    parser.add_argument('-v', '--select_views', nargs='*', help="List of views that should be used.")
    parser.add_argument('-s', '--scale', type=float, default=0.25,
                        help="Rescale the images to be processed. Defaults is 0.25 to reduce runtime. " +
                             "Use 1.0 for full resolution.")
    parser.add_argument('-n', '--num_points', type=int, default=10 ** 6,
                        help="Number of points to sample from the mesh. Default is 1,000,000")
    args = parser.parse_args()

    run(args.obj_or_ply_path, args.cameras_path, args.images_dir,
        args.out_dir, args.select_views, args.num_points, args.scale)
