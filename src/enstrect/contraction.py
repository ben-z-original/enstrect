from tqdm import tqdm
import numpy as np
import networkx as nx
from pc_skeletor import LBC
from pyntcloud import PyntCloud
from pathlib import Path
import itertools
from scipy.spatial.distance import cdist
import open3d as o3d


def subcloud2polyline(subcloud, polyline_path, cluster_id):
    # Init tree skeletonizer

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

    print()
    polyline2obj(G, polyline_path, cluster_id)


def polyline2obj(G, polyline_path, cluster_id):
    # prepare line obj
    out = ""
    nx.relabel_nodes(G, {key: i for i, key in enumerate(G.nodes)}, copy=False)
    # positions = nx.get_node_attributes(G, "pos")
    positions = np.zeros((len(G), 3))
    for i in range(len(G)):
        node_pos = G.nodes[i]['pos']
        out += f"v {node_pos[0]} {node_pos[1]} {node_pos[2]}\n"
        positions[i, ...] = node_pos

    line_geoms = []

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for SG in S:
        degrees = np.array([val for (node, val) in SG.degree()])
        src, trg = np.where(degrees == 1)[0]
        path = nx.shortest_path(G, source=list(SG.nodes)[src], target=list(SG.nodes)[trg])
        path = np.int64(np.array(path)) + 1
        out += "l"
        for elem in path:
            out += f" {elem}"
        out += "\n"

        lines = [[path[i] - 1, path[i + 1] - 1] for i in range(len(path) - 1)]
        # lines.append([boundary_idxs[-1], boundary_idxs[0]])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # line_geoms.extend(lineset2linemesh(line_set, color=[1, 0, 0]))

    # o3d.visualization.draw_geometries([*line_geoms, mesh])  # subcloud])

    with open(str(polyline_path / f"crack_{cluster_id:04d}.obj"), "w") as f:
        f.write(out)


if __name__ == "__main__":
    clustered_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud_tmp2.ply"
    cloud = PyntCloud.from_file(str(clustered_path))

    defect = np.array(cloud.points["defect"])
    cluster = np.array(cloud.points["cluster"])
    cluster_ids, cluster_counts = np.unique(cluster, return_counts=True)
    classes = {c: defect[cluster == c][0] for c in cluster_ids}

    cloud = cloud.to_instance("open3d", mesh=False, normals=True)
    cloud.paint_uniform_color([0.5, 0.5, 0.5])

    for i, cluster_id in tqdm(enumerate(cluster_ids[1:])):
        idxs = np.nonzero(cluster == cluster_id)[0]

        # crack
        if classes[cluster_id] == 1:  # and np.median(cluster_counts) <= cluster_counts[
            # cluster_id]:  # TODO: selection more generic
            subcloud = cloud.select_by_index(idxs)
            subcloud2polyline(subcloud, Path("/media/chrisbe/backup/segments/bridge_b/segment1/cloud/"),
                              cluster_id)
