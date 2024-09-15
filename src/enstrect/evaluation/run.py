import numpy as np
import open3d as o3d
import networkx as nx
from pathlib import Path
from utils3d.conversion.lines import obj_to_G, G_to_lineset_o3d, G_to_linemesh_o3d, interpolate_G
from enstrect.evaluation.metrics import G_to_lineIoU3d


if __name__ == "__main__":
    vis = True

    structure = "bridge_g"#, "bridge_s", ]:
    segment = "segment2" #"segment1", "segment2", "segment3"]:
    damages = ["spalling", "exposed_rebar"]

    rootdir = Path(
        f"/home/chrisbe/repos/dissertation/enstrect/src/enstrect/assets/crackstructures/{structure}/{segment}")

    G_true = nx.Graph()
    G_pred = nx.Graph()

    for damage in damages:
        if (rootdir / "annotations" / f"{damage}.obj").exists():
            G_true = nx.compose(G_true, obj_to_G(rootdir / "annotations" / f"{damage}.obj"))
        if (rootdir / "out" / f"{damage}.obj").exists():
            G_pred = nx.compose(G_pred, obj_to_G(rootdir / "out" / f"{damage}.obj"))

    tolerances = [0.01, 0.02, 0.04, 0.06, 0.08]

    ious = []

    for tol in tolerances:
        gap = tol / 4
        iou = G_to_lineIoU3d(G_true, G_pred, gap, tol)[0]
        ious.append(iou)

    tex = ""
    for iou in ious:
        tex += f" & {100*iou:.1f}"

    #with open(f"/home/chrisbe/dissertation/tables/crackstructures/positional_tolerance/{structure}_{segment}.tex", 'w') as f:
    #    f.write(tex)
    print(tex)

    if vis:
        linemesh_true = G_to_linemesh_o3d(G_true, color=[0, 0, 1])
        linemesh_pred = G_to_linemesh_o3d(G_pred, color=[1, 0, 0], radius=0.001)

        mesh = o3d.io.read_triangle_mesh(str(rootdir / "mesh" / "mesh.obj"), True)
        o3d.visualization.draw_geometries([*linemesh_pred, mesh]) #*linemesh_true

    print()
