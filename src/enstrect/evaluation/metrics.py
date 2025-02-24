import torch
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import distance_matrix
from pytorch3d.loss import chamfer_distance

def G_to_clCloudIoU(G_true, G_pred, gap=0.001, tol=0.01):
    H_true = interpolate_G(G_true, gap=gap)
    H_pred = interpolate_G(G_pred, gap=gap)

    true_pos = np.array([data["pos"] for _, data in list(H_true.nodes(data=True))])
    pred_pos = np.array([data["pos"] for _, data in list(H_pred.nodes(data=True))])

    clcloudiou = clCloudIoU(true_pos, pred_pos, tol=tol)
    return clcloudiou

def clCloudIoU(true_pos, pred_pos, tol=0.001):
    """Computes the centerline intersection-over-union on lines in 3D space based on a tolerance value."""
    if len(pred_pos) == 0 and len(true_pos) == 0:
        return 1, 0, 0, 0
    elif len(true_pos) == 0:
        return 0, 0, 0, len(pred_pos)
    elif len(pred_pos) == 0:
        return 0, 0, len(true_pos), 0

    # determine true/false positives/negatives
    distances = distance_matrix(true_pos, pred_pos)
    tp = np.sum(np.min(distances, axis=1) <= tol)
    fn = np.sum(np.min(distances, axis=1) > tol)
    fp = np.sum(np.min(distances, axis=0) > tol)
    cl_cloud_iou = tp / (tp + fn + fp)

    return cl_cloud_iou, tp, fn, fp


def interpolate_G(G, gap=0.01):
    S = [nx.convert_node_labels_to_integers(G.subgraph(c), first_label=0) for c in nx.connected_components(G)]
    H = nx.Graph()

    for SG in S:
        if len(SG) == 0:
            continue
        positions = np.array([SG.nodes[node_id]['pos'] for node_id in SG.nodes])
        degrees = np.array([val for (node, val) in SG.degree()])
        if np.all(degrees < 1) or np.all(2 < degrees):  # TODO: for boundaries different
            raise RuntimeError("Graph contains branches")
        elif np.all(2 == degrees):
            # remove arbitrary edge
            SG.remove_edge(*list(G.edges())[0])
            degrees = np.array([val for (node, val) in SG.degree()])
        try:
            src, trg = np.where(degrees == 1)[0]
        except:
            print()
            print()
            continue
        path = np.array(nx.shortest_path(SG, source=list(SG.nodes)[src], target=list(SG.nodes)[trg]))

        line_pos = positions[path]
        line_inter = interpolate_line_points(line_pos, gap=gap)

        size_prev = len(H)
        nx.add_path(H, np.arange(len(line_inter)) + size_prev)
        nx.set_node_attributes(H, {i + size_prev: pos for i, pos in enumerate(line_inter)}, "pos")

    return H

def interpolate_line_points(pts, gap=0.01):
    diff = np.sqrt(np.sum(np.power(np.diff(pts, axis=0), 2), axis=1))
    xp = np.append(0, np.cumsum(diff))
    x = np.arange(0 + xp[-1] % gap / 2, xp[-1], gap)
    x = np.hstack((0, x, xp[-1]))

    # interpolate points
    points = np.stack([np.interp(x, xp, pts[:, 0]),
                       np.interp(x, xp, pts[:, 1]),
                       np.interp(x, xp, pts[:, 2])], axis=1)

    return points
