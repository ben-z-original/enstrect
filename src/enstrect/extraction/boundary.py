import alphashape
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import PCA
from enstrect.extraction.clustering import get_pcd_classes


def extract_bounding_polygons(pcd_pynt, classes=["corrosion"], min_count=2000):
    defect = np.array(pcd_pynt.points["argmax"])
    cluster = np.array(pcd_pynt.points["cluster"])
    pcd_o3d = pcd_pynt.to_instance("open3d", mesh=False, normals=True)

    # get classes of clusters
    pcd_classes = get_pcd_classes(pcd_pynt)
    cluster_ids, cluster_counts = np.unique(cluster, return_counts=True)
    cluster_classes = {clu_id: pcd_classes[defect[cluster == clu_id][0]] for clu_id in cluster_ids}

    pca = PCA(n_components=2)
    G = nx.Graph()

    for cla in classes:
        relevant_clusters = [key for (key, value) in cluster_classes.items() if value == cla]

        for cluster_id in tqdm(relevant_clusters):
            if min_count < cluster_counts[cluster_id]:
                idxs = np.nonzero(cluster == cluster_id)[0]
                subcloud = pcd_o3d.select_by_index(idxs)

                # extract boundary
                points = np.array(subcloud.points)
                points_mapped = pca.fit_transform(points)
                alpha_shape = alphashape.alphashape(points_mapped, 100)
                bound_coords = np.array(alpha_shape.boundary.coords)
                bound_idxs = [np.where((points_mapped == bound_coords[i, ...]).all(axis=1))[0][0]
                              for i in range(len(bound_coords))]
                bound_points = points[bound_idxs[:-1]]

                # prepare graph
                H = nx.Graph()
                edges = [(i - 1, i) for i in range(1, len(bound_points))] + [(len(bound_points) - 1, 0)]
                H.add_edges_from(edges)
                nx.set_node_attributes(H, {i: bound_points[i] for i in range(len(bound_points))}, "pos")

                H = nx.relabel_nodes(H, {key: i + len(G) for i, key in enumerate(H.nodes)}, copy=True)
                G = nx.compose(G, H)

    return G
