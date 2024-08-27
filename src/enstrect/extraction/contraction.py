import itertools
import numpy as np
import open3d as o3d
from tqdm import tqdm
import networkx as nx
from pathlib import Path
from pc_skeletor import LBC
from pyntcloud import PyntCloud
from scipy.spatial.distance import cdist
from utils3d.conversion.lines import polyline2obj
from utils3d.visualization.visualize_o3d import plot_o3d_mesh_with_lines


def extract_centerlines(pcd_pynt):
    defect = np.array(pcd_pynt.points["argmax"])
    cluster = np.array(pcd_pynt.points["cluster"])
    cluster_ids, cluster_counts = np.unique(cluster, return_counts=True)
    classes = {c: defect[cluster == c][0] for c in cluster_ids}

    cloud = pcd_pynt.to_instance("open3d", mesh=False, normals=True)
    cloud.paint_uniform_color([0.5, 0.5, 0.5])

    G = nx.Graph()
    for i, cluster_id in tqdm(enumerate(cluster_ids[1:])):
        idxs = np.nonzero(cluster == cluster_id)[0]

        # crack
        if classes[cluster_id] == 1:
            subcloud = cloud.select_by_index(idxs)
            # o3d.visualization.draw_geometries([subcloud])
            try:
                H = contract_subcloud(subcloud)

                H = nx.relabel_nodes(H, {key: i + len(G) for i, key in enumerate(H.nodes)}, copy=True)
                G = nx.compose(G, H)
            except:
                raise UserWarning(f"Contraction failed for cluster ID {cluster_id}")

    return G


def contract_subcloud(subcloud):
    # init Laplacian-based contraction
    lbc = LBC(point_cloud=subcloud,
              down_sample=0.002,
              init_contraction=5.,
              init_attraction=1.,
              max_contraction=2048,
              max_attraction=1024,
              step_wise_contraction_amplification='auto',
              termination_ratio=0.003,
              max_iteration_steps=20,
              filter_nb_neighbors=20,
              filter_std_ratio=2.0,
              )
    lbc.extract_skeleton()

    points = np.array(lbc.contracted_point_cloud.points)

    # prepare fully connected graph with distances as edge weight and position as node attribute
    G = nx.Graph()
    distances = cdist(points, points)
    combinations = np.array(list(itertools.combinations(range(len(points)), 2)))
    entries = np.append(combinations, distances[combinations[:, 0], combinations[:, 1]][:, None], axis=-1)
    G.add_weighted_edges_from(entries)
    nx.set_node_attributes(G, {i: points[i] for i in range(len(points))}, "pos")

    # compute minimal spanning tree
    G = nx.minimum_spanning_tree(G)
    degrees = np.array([val for (node, val) in G.degree()])
    furcations = np.where(2 < degrees)[0]

    # partition the tree
    for furc_node in furcations:
        neighs = list(G.neighbors(furc_node))
        for i, neigh in enumerate(neighs):
            G.add_node(np.max(G.nodes) + 1, pos=G.nodes[furc_node]['pos'])
            G.add_edge(np.max(G.nodes), neigh)
            G.remove_edge(furc_node, neigh)
        G.remove_node(furc_node)

    return G


if __name__ == "__main__":
    clustered_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud_tmp2.ply"
    cloud = PyntCloud.from_file(str(clustered_path))

    G = extract_centerlines(cloud)

    polyline2obj(G, Path("/media/chrisbe/backup/segments/bridge_b/segment1/cloud/"), 0)
    mesh = o3d.io.read_triangle_mesh("/media/chrisbe/backup/segments/bridge_b/segment1/model.obj", True)
    plot_o3d_mesh_with_lines(mesh, G)
