import warnings
import itertools
import numpy as np
from tqdm import tqdm
import networkx as nx
from pc_skeletor import LBC
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


def extract_centerlines(pcd_pynt, category="crack", eps_m=0.01, min_points=20):
    cloud = pcd_pynt.to_instance("open3d", mesh=False, normals=True)

    # subcloud from category
    idxs = np.nonzero(pcd_pynt.points[category])[0]
    points_pd = pcd_pynt.points.loc[idxs].reset_index(drop=False)
    points_np = np.array(points_pd[['x', 'y', 'z']], dtype=np.float64)

    G = nx.Graph()
    if len(points_np) == 0: return G

    # cluster analysis
    cluster = DBSCAN(eps=eps_m, min_samples=20).fit_predict(points_np)
    cluster_ids, cluster_counts = np.unique(cluster, return_counts=True)
    cluster_ids, cluster_counts = cluster_ids[cluster_ids != -1], cluster_counts[cluster_ids != -1]

    for (cluster_id, cluster_count) in tqdm(zip(cluster_ids, cluster_counts)):
        if cluster_count < min_points: continue

        # subcloud of current cluster
        idxs_cluster = idxs[np.nonzero(cluster == cluster_id)[0]]
        subcloud = cloud.select_by_index(idxs_cluster)
        try:
            H = contract_subcloud(subcloud)
            H = nx.relabel_nodes(H, {key: i + len(G) for i, key in enumerate(H.nodes)}, copy=True)
            G = nx.compose(G, H)
        except:
            warnings.warn(f"Contraction failed for cluster ID {cluster_id}")
            #print(f"Contraction failed for cluster ID {cluster_id}")#raise UserWarning(f"Contraction failed for cluster ID {cluster_id}")

    return G


def contract_subcloud(subcloud):
    # init Laplacian-based contraction
    lbc = LBC(point_cloud=subcloud,
              down_sample=0.002,
              init_contraction=1.,
              init_attraction=1.,
              max_contraction=256, #2048,
              max_attraction=2048, #1024,
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
