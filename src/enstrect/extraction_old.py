import shapely
import itertools
import alphashape
import numpy as np
import open3d as o3d
from tqdm import tqdm
import networkx as nx
from pathlib import Path
from pyntcloud import PyntCloud
#from enstrect.config import cfg
from pc_skeletor import skeletor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
#from enstrect.utils.line import lineset2linemesh


def cluster_point_cloud(ply_path, out_path, eps=0.01):
    """Clusters the defects of a point cloud."""
    ply_pred = PyntCloud.from_file(str(ply_path))
    ply_pred.points["cluster"] = np.zeros((len(ply_pred.points),), np.ubyte)  # empty attribute

    for k in range(1, 4):
        points = ply_pred.points.loc[ply_pred.points["defect"] == k].reset_index(drop=False)
        if len(points) == 0:
            continue
        cluster = DBSCAN(eps=eps, min_samples=20).fit_predict(np.array(points[['x', 'y', 'z']]))

        ply_pred.points.loc[points["index"], "cluster"] = \
            (cluster + ply_pred.points["cluster"].max() + 1).astype(np.ubyte)
        ply_pred.points.loc[points["index"][cluster == -1], "cluster"] = 0

        ply_pred.to_file(str(out_path))


def line2obj(points, boundary_idxs=None, path=None, polygon=True):
    """Converts the 3D boundary points to line obj."""
    if boundary_idxs is None:
        boundary_idxs = np.arange(len(points))  # all vertices!

    out = ""
    for vert in points[boundary_idxs]:
        out += f"v {vert[0]} {vert[1]} {vert[2]}\n"
    out += "l"
    for i in range(len(boundary_idxs)):
        out += f" {i + 1}"

    if polygon:
        out += " 1"  # close polygon

    if path is not None:
        with open(str(path), "w") as f:
            f.write(out)
    return out


def extract_bounding_polygon(mapped_points):
    """Computes the bounding polygon of a point cloud which is mapped to 2D space."""
    alpha_shape = alphashape.alphashape(mapped_points, 100)

    boundary_idxs = []

    alpha_shape = [alpha_shape] if not isinstance(alpha_shape,
                                                  shapely.geometry.multipolygon.MultiPolygon) else alpha_shape

    for i in range(len(alpha_shape)):
        bound = np.array(alpha_shape[i].boundary.coords)

        boundary_idxs.append([np.where((mapped_points == bound[i, ...]).all(axis=1))[0][0] for i in range(len(bound))])

        # 2d visualization
        if False:
            plt.figure(figsize=(20, 12))
            plt.plot(mapped_points[:, 0], mapped_points[:, 1], '.', color="darkgray")
            plt.plot(bound[:, 0], bound[:, 1], color=[1, 0.5, 0], linewidth=5)  # , 'o')
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            # plt.savefig("/home/chrisbe/Desktop/tmp2.png")
            plt.show()

    return boundary_idxs


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
        line_geoms.extend(lineset2linemesh(line_set, color=[1, 0, 0]))

    # o3d.visualization.draw_geometries([*line_geoms, mesh])  # subcloud])

    with open(str(polyline_path / f"crack_{cluster_id:04d}.obj"), "w") as f:
        f.write(out)


def subcloud2polyline(subcloud, polyline_path, cluster_id):
    # Init tree skeletonizer
    skeletonize = skeletor.Skeletonizer(point_cloud=subcloud,
                                        down_sample=0.0005,
                                        debug=False)
    laplacian_config = {"MAX_LAPLACE_CONTRACTION_WEIGHT": 2 ** 2,
                        "MAX_POSITIONAL_WEIGHT": 2 ** 13,
                        "INIT_LAPLACIAN_SCALE": 100}
    _, skeleton = skeletonize.extract(method='Laplacian', config=laplacian_config)

    # prepare fully connected graph with distances as edge weight and position as node attribute
    G = nx.Graph()
    distances = cdist(skeleton.points, skeleton.points)
    combinations = np.array(list(itertools.combinations(range(len(skeleton.points)), 2)))
    entries = np.append(combinations, distances[combinations[:, 0], combinations[:, 1]][:, None], axis=-1)
    G.add_weighted_edges_from(entries)
    nx.set_node_attributes(G, {i: skeleton.points[i] for i in range(len(skeleton.points))}, "pos")

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

    polyline2obj(G, polyline_path, cluster_id)


def subcloud2polygon(subcloud, polygon_path, name, cluster_id):
    """Computes bounding polygon for areal defects."""
    # map to 2D
    points = np.array(subcloud.points)
    pca = PCA(n_components=2)
    mapped_points = pca.fit_transform(points)

    # extract bounding polygon
    boundary_idxs = extract_bounding_polygon(mapped_points)

    for bound in boundary_idxs:
        # open3d visualization
        # visualize_in_open3d(subcloud, bound)

        # obj export
        line2obj(points, bound, path=polygon_path / f"{name}_{cluster_id:04d}.obj")


def extract_defects(clustered_path, extracted_path):
    """Extracts cracks and areal defects from a clustered point cloud."""
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
        if classes[cluster_id] == 1:# and np.median(cluster_counts) <= cluster_counts[
            #cluster_id]:  # TODO: selection more generic
            subcloud = cloud.select_by_index(idxs)
            subcloud2polyline(subcloud, extracted_path, cluster_id)
        # spalling
        elif classes[cluster_id] == 2 and 2000 < cluster_counts[cluster_id]:
            subcloud = cloud.select_by_index(idxs)
            subcloud2polygon(subcloud, extracted_path, "spalling", cluster_id)
        # corrosion
        elif classes[cluster_id] == 3 and 1500 < cluster_counts[cluster_id]:
            subcloud = cloud.select_by_index(idxs)
            subcloud2polygon(subcloud, extracted_path, "corrosion", cluster_id)


if __name__ == "__main__":

    mesh = o3d.io.read_triangle_mesh("/home/chrisbe/Desktop/crackensembles/00000/mesh_color_0.png")
    o3d.visualization.draw_geometries([mesh])

    ply_path = cfg.root / cfg.mode / "colorized" / cfg.case / cfg.detector / cfg.fusor / "colorized.ply"
    out_path = ply_path
    extracted_path = cfg.root / cfg.mode / "extracted" / cfg.case / cfg.detector / cfg.fusor
    cluster_point_cloud(ply_path, out_path)
    extract_defects(out_path, extracted_path)
