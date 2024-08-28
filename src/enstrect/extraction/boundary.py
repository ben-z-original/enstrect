import alphashape
import numpy as np
import open3d as o3d
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def extract_bounding_polygons(pcd_pynt, category="corrosion", eps_m=0.01, min_count=2000):
    points_pd = pcd_pynt.points.loc[pcd_pynt.points[category] == 1].reset_index(drop=False)
    points_np = np.array(points_pd[['x', 'y', 'z']], dtype=np.float64)

    cluster = DBSCAN(eps=eps_m, min_samples=20).fit_predict(points_np)
    cluster_ids, cluster_counts = np.unique(cluster, return_counts=True)

    pca = PCA(n_components=2)
    G = nx.Graph()

    for cluster_id in tqdm(cluster_ids):
        idxs = np.nonzero(cluster == cluster_id)[0]

        points_subcloud = points_np[idxs]

        if False:
            print()
            pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_subcloud))
            o3d.visualization.draw_geometries([pcd_o3d])
            #subcloud = pcd_o3d.select_by_index(idxs)

        # extract boundary
        points_mapped = pca.fit_transform(points_subcloud)
        alpha_shape = alphashape.alphashape(points_mapped, 100)
        bound_coords = np.array(alpha_shape.boundary.coords)
        bound_idxs = [np.where((points_mapped == bound_coords[i, ...]).all(axis=1))[0][0]
                      for i in range(len(bound_coords))]
        bound_points = points_subcloud[bound_idxs[:-1]]

        # prepare graph
        H = nx.Graph()
        edges = [(i - 1, i) for i in range(1, len(bound_points))] + [(len(bound_points) - 1, 0)]
        H.add_edges_from(edges)
        nx.set_node_attributes(H, {i: bound_points[i] for i in range(len(bound_points))}, "pos")

        H = nx.relabel_nodes(H, {key: i + len(G) for i, key in enumerate(H.nodes)}, copy=True)
        G = nx.compose(G, H)

    return G
