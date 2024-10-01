import torch
import numpy as np
import pandas as pd
import open3d as o3d
import networkx as nx
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN
from pytorch3d.structures import Pointclouds


def prepare_exposed_rebar(pcd_pynt, eps_m=0.01):
    if not "spalling" in pcd_pynt.points.columns or not "corrosion" in pcd_pynt.points.columns:
        return pcd_pynt

    pcd_pynt.points["exposed_rebar"] = pcd_pynt.points["spalling"] + pcd_pynt.points["corrosion"]
    idxs = np.nonzero(pcd_pynt.points["exposed_rebar"])[0]

    if len(idxs) == 0:
        return pcd_pynt

    # cluster exposed rebar
    points_pd = pcd_pynt.points.loc[idxs].reset_index(drop=False)
    cluster = DBSCAN(eps=eps_m, min_samples=20).fit_predict(np.array(points_pd[['x', 'y', 'z']]))
    np.unique(cluster)

    for i in np.unique(cluster):
        idxs_cluster = idxs[cluster == i]

        spalling_count = np.sum(pcd_pynt.points["spalling"][idxs_cluster])
        corrosion_count = np.sum(pcd_pynt.points["corrosion"][idxs_cluster])
        exposed_rebar_count = np.sum(pcd_pynt.points["exposed_rebar"][idxs_cluster])

        # pure spalling
        if spalling_count == len(idxs_cluster):
            pcd_pynt.points.loc[idxs_cluster, "exposed_rebar"] = 0
        # pure corrosion
        elif corrosion_count == len(idxs_cluster):
            pcd_pynt.points.loc[idxs_cluster, "exposed_rebar"] = 0
        # exposed rebar (=combination of spalling and corrosion)
        else:
            pcd_pynt.points.loc[idxs_cluster, "exposed_rebar"] = 1
            pcd_pynt.points.loc[idxs_cluster, "spalling"] = 0
            # pcd_pynt.points.loc[idxs_cluster, "corrosion"] = 0  # keep corrosion as is

    return pcd_pynt


def pyt3d_to_pynt(pcd_pyt3d):
    points_pd = pd.DataFrame(torch.cat([pcd_pyt3d.points_packed().squeeze(),
                                        pcd_pyt3d.normals_packed().squeeze(),
                                        pcd_pyt3d.features_packed().squeeze()], axis=1),
                             columns=["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"])
    pcd_pynt = PyntCloud(points_pd)
    return pcd_pynt

def pynt_to_pyt3d(pcd_pynt, device="cuda:0"):
    pcd_pyt3d = Pointclouds(torch.tensor(pcd_pynt.points[["x", "y", "z"]].to_numpy()[None, ...],
                                         dtype=torch.float32, device=device),
                            torch.tensor(pcd_pynt.points[["nx", "ny", "nz"]].to_numpy()[None, ...],
                                         dtype=torch.float32, device=device),
                            torch.tensor(pcd_pynt.points[["red", "green", "blue"]].to_numpy()[None, ...],
                                         dtype=torch.float32, device=device))
    return pcd_pyt3d

def G_to_obj(G, polyline_path):
    if len(G.nodes) == 0: return None

    line_sets = G_to_lineset_o3d(G)

    vertices_obj, lines_obj = "", ""
    count = 1  # obj starts counting at 1

    for line_set in line_sets:
        vertices_obj += "".join([f"v {x} {y} {z}\n" for x, y, z in np.array(line_set.points)])
        lines = np.array(line_set.lines).flatten()
        lines = np.append(lines[[0]], lines[1:][np.diff(lines) != 0]) + count
        lines_obj += "l " + " ".join(map(str, lines)) + "\n"
        count += len(line_set.points)

    with open(str(polyline_path), "w") as f:
        f.write(vertices_obj + lines_obj)


def G_to_lineset_o3d(G, color=[1, 0, 0]):
    # connected subgraphs
    S = [nx.convert_node_labels_to_integers(G.subgraph(c), first_label=0) for c in nx.connected_components(G)]

    line_sets = []
    for SG in S:
        positions = np.array([SG.nodes[node_id]['pos'] for node_id in SG.nodes])
        degrees = np.array([val for (node, val) in SG.degree()])

        # is polygon (boundary)
        if np.all(1 < degrees):
            cycle_edges = nx.find_cycle(SG)
            lines = np.array(cycle_edges)
        # is polyline (centerline)
        else:
            src, trg = np.where(degrees == 1)[0]
            path = nx.shortest_path(SG, source=list(SG.nodes)[src], target=list(SG.nodes)[trg])
            path = np.int64(np.array(path)) + 1
            lines = [[path[i] - 1, path[i + 1] - 1] for i in range(len(path) - 1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        line_sets.append(line_set)

    return line_sets
