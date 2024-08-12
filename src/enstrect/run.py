import torch
import open3d as o3d
from pathlib import Path
from pyntcloud import PyntCloud
from argparse import ArgumentParser
from utils3d.fusion.static import MeanFuser
from enstrect.contraction import extract_lines
from utils3d.conversion.lines import polyline2obj
from enstrect.clustering import cluster_point_cloud
from enstrect.semantic_mapper import SemanticMapper
from omnicrack30k.inference import OmniCrack30kModel
from utils3d.datasets.multiview_dataset import MultiviewDataset
from utils3d.visualization.visualize_o3d import plot_o3d_mesh_with_lines

if __name__ == "__main__":
    parser = ArgumentParser(description="""Run enstrect.""")
    parser.add_argument('-p', '--ply_path', type=str, default="/media/chrisbe/backup/segments/bridge_b/segment2/cloud/cloud.ply", help="")
    parser.add_argument('-i', '--images_dir', type=str, default="/media/chrisbe/backup/segments/bridge_b/segment2/real/original", help="")
    parser.add_argument('-c', '--cameras_xml_path', type=str, default="/media/chrisbe/backup/segments/bridge_b/segment2/real/cameras.xml", help="")
    parser.add_argument('-po', '--ply_out_path', type=str, default="/media/chrisbe/backup/segments/bridge_b/segment2/cloud/cloud_tmp.ply", help="")
    parser.add_argument('-pp', '--polyline_path', type=str, default="/media/chrisbe/backup/segments/bridge_b/segment2/cloud/cloud_tmp.obj", help="")
    args = parser.parse_args()

    #ply_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud.ply"
    #images_dir = "/media/chrisbe/backup/segments/bridge_b/segment1/real/original"
    #cameras_xml_path = "/media/chrisbe/backup/segments/bridge_b/segment1/real/cameras.xml"
    #ply_out_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud_tmp.ply"
    #polyline_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud_tmp.obj"

    pcd_pynt = PyntCloud.from_file(str(args.ply_path))
    model = OmniCrack30kModel()
    dataset = MultiviewDataset(args.cameras_xml_path, args.images_dir)
    fuser = MeanFuser(class_weight=torch.tensor([1, 5], dtype=torch.float16))
    mapper = SemanticMapper(model, fuser=fuser)

    pcd_pynt = mapper(pcd_pynt, dataset)
    pcd_pynt = cluster_point_cloud(pcd_pynt)
    pcd_pynt.to_file(args.ply_out_path)

    G = extract_lines(pcd_pynt)

    polyline2obj(G, args.polyline_path)
    mesh = o3d.io.read_triangle_mesh("/media/chrisbe/backup/segments/bridge_b/segment1/model.obj", True)
    plot_o3d_mesh_with_lines(mesh, G)

