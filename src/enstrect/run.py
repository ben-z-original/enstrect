import open3d as o3d
from pathlib import Path
from pyntcloud import PyntCloud
from argparse import ArgumentParser
from enstrect.mapping.fuser import Fuser
from utils3d.conversion.lines import polyline2obj
from enstrect.mapping.mapper import Mapper
from enstrect.segmentation.nnunet_s2ds import NNUNetS2DSModel
from enstrect.extraction.clustering import cluster_point_cloud
from enstrect.extraction.contraction import extract_centerlines
from utils3d.datasets.multiview_dataset import MultiviewDataset
from enstrect.extraction.boundary import extract_bounding_polygons
from utils3d.visualization.visualize_o3d import plot_o3d_mesh_with_lines

if __name__ == "__main__":
    parser = ArgumentParser(description="""Run enstrect.""")
    parser.add_argument('-p', '--ply_path', type=str,
                        default=str(Path(__file__).parents[0] / "assets" / "example_exposed_rebar" / "cloud.ply"),
                        help="Path to the point cloud file in ply format.")
    parser.add_argument('-i', '--images_dir', type=str,
                        default=str(Path(__file__).parents[0] / "assets" / "example_exposed_rebar" / "views"),
                        help="Path to the directory which contains the images/views.")
    parser.add_argument('-c', '--cameras_path', type=str,
                        default=str(Path(__file__).parents[0] / "assets" / "example_exposed_rebar" / "cameras.xml"),
                        help="Path to the file that contains the intrinsic and extrinsic camera information")
    parser.add_argument('-o', '--out_dir', type=str,
                        default=str(Path(__file__).parents[0] / "assets" / "example_exposed_rebar" / "out"),
                        help="Path to the directory where the results will be stored")
    args = parser.parse_args()

    # set up segmentation model, mapper, and dataset
    model = NNUNetS2DSModel()
    fuser = Fuser(class_weight=model.class_weight)
    mapper = Mapper(model, fuser=fuser)
    dataset = MultiviewDataset(args.cameras_path, args.images_dir)

    # perform mapping 2D->3D
    pcd_pynt = PyntCloud.from_file(str(args.ply_path))
    pcd_pynt = mapper(pcd_pynt, dataset)
    pcd_pynt = cluster_point_cloud(pcd_pynt)
    pcd_pynt.to_file(str(Path(args.out_dir) / "cloud_processed.ply"))

    # perform extraction
    #G = extract_centerlines(pcd_pynt)
    G = extract_bounding_polygons(pcd_pynt)

    #polyline2obj(G, str(Path(args.out_dir) / "lines.obj"))
    mesh = o3d.io.read_triangle_mesh("/media/chrisbe/backup/segments/bridge_g/segment3/model.obj", True)
    plot_o3d_mesh_with_lines(mesh, G)
