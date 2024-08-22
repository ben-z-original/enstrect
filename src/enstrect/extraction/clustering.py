import numpy as np
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN


def cluster_point_cloud(pcd_pynt, eps_m=0.01):
    """Clusters the defects of a point cloud."""
    pcd_pynt.points["cluster"] = np.zeros((len(pcd_pynt.points),), np.ubyte)  # empty attribute

    for k in np.unique(pcd_pynt.points["argmax"])[1:]:  # 0 is assumed to be background
        points = pcd_pynt.points.loc[pcd_pynt.points["argmax"] == k].reset_index(drop=False)
        cluster = DBSCAN(eps=eps_m, min_samples=20).fit_predict(np.array(points[['x', 'y', 'z']]))

        pcd_pynt.points.loc[points["index"], "cluster"] = \
            (cluster + pcd_pynt.points["cluster"].max() + 1).astype(np.ubyte)
        pcd_pynt.points.loc[points["index"][cluster == -1], "cluster"] = 0

    return pcd_pynt


def get_pcd_classes(pcd_pynt):
    removables = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "argmax", "cluster"]
    classes = [elem for elem in list(pcd_pynt.points.columns) if elem not in removables]
    return classes


if __name__ == "__main__":
    ply_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud_tmp.ply"
    ply_out_path = "/media/chrisbe/backup/segments/bridge_b/segment1/cloud/cloud_tmp2.ply"

    ply_pred = PyntCloud.from_file(str(ply_path))
    pcd_pynt = cluster_point_cloud(ply_pred, None)
    pcd_pynt.to_file(ply_out_path)
