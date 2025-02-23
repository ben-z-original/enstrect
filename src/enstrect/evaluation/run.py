import argparse
import numpy as np
import open3d as o3d
import networkx as nx
from pathlib import Path
from enstrect.evaluation.metrics import G_to_clCloudIoU
from enstrect.evaluation.utils.lines import obj_to_G, G_to_lineset_o3d, G_to_linemesh_o3d, interpolate_G


def evaluate(segment_path, damages, vis=True, tolerances=[0.005, 0.01, 0.02, 0.04, 0.08],
             tex_path=None):

    # damages to list if only one damage is given
    damages = [damages] if not isinstance(damages, list) else damages

    # prepare graphs
    G_true, G_pred = nx.Graph(), nx.Graph()
    for damage in damages:
        true_path = segment_path / "annotations" / f"{damage}.obj"
        pred_path = segment_path / "out" / f"{damage}.obj"

        G_true = nx.compose(G_true, obj_to_G(true_path)) if true_path.exists() else G_true
        G_pred = nx.compose(G_pred, obj_to_G(pred_path)) if pred_path.exists() else G_pred

    # compute ious
    ious = []
    for tol in tolerances:
        iou = G_to_clCloudIoU(G_true, G_pred, tol=tol)[0]
        ious.append(iou)

    # console output
    [print(f"{tol} \t{iou:.3f}") for tol, iou in zip(tolerances, ious)]

    if tex_path is not None:
        tex = ""
        for iou in ious:
            tex += f" & {100 * iou:.1f}"
        print(tex)

        with open(str(tex_path), 'w') as f:
            f.write(tex)

    # visualize
    if vis:
        linemesh_true = G_to_linemesh_o3d(G_true, color=[0, 0, 1])
        linemesh_pred = G_to_linemesh_o3d(G_pred, color=[1, 0, 0], radius=0.001)

        mesh = o3d.io.read_triangle_mesh(str(segment_path / "mesh" / "mesh.obj"), True)
        o3d.visualization.draw_geometries([*linemesh_pred, mesh])  # *linemesh_true


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize results.")
    parser.add_argument('--structure', type=str, default="bridge_g", help="Structure type")
    parser.add_argument('--segment', type=str, default="segment_test", help="Segment type")
    parser.add_argument('--damage', nargs='+', default=["spalling", "exposed_rebar"],
                        help="Structural damage (or list of structural damages, in case they form a combined class")
    parser.add_argument('--datadir', type=Path,
                        default=Path(__file__).parents[1] / "assets" / "segments",
                        help="Directory path to dataset root")
    parser.add_argument('--vis', action='store_true', help="Enable visualization")
    args = parser.parse_args()

    segment_path = args.datadir / args.structure / args.segment

    evaluate(segment_path, args.damage, vis=args.vis, tex_path=None)
