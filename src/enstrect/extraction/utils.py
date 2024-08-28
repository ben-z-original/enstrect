import numpy as np
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN


def prepare_exposed_rebar(pcd_pynt, eps_m=0.01):
    pcd_pynt.points["exposed_rebar"] = pcd_pynt.points["spalling"] + pcd_pynt.points["corrosion"]
    idxs = np.where(pcd_pynt.points["exposed_rebar"] == 1)[0]

    # cluster exposed rebar
    points = pcd_pynt.points.loc[idxs].reset_index(drop=False)
    cluster = DBSCAN(eps=eps_m, min_samples=20).fit_predict(np.array(points[['x', 'y', 'z']]))
    np.unique(cluster)

    for i in np.unique(cluster):
        idxs_cluster = idxs[cluster == i]

        spalling_count = np.sum(pcd_pynt.points["spalling"][idxs_cluster])
        corrosion_count = np.sum(pcd_pynt.points["corrosion"][idxs_cluster])
        exposed_rebar_count = np.sum(pcd_pynt.points["exposed_rebar"][idxs_cluster])

        # readjust classes
        pcd_pynt.points.loc[idxs_cluster, "spalling"] = np.ubyte(spalling_count == len(idxs_cluster))
        pcd_pynt.points.loc[idxs_cluster, "corrosion"] = np.ubyte(corrosion_count == len(idxs_cluster))
        pcd_pynt.points.loc[idxs_cluster, "exposed_rebar"] = np.ubyte(exposed_rebar_count == len(idxs_cluster))

    return pcd_pynt


def get_pcd_classes(pcd_pynt):
    removables = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "argmax", "cluster"]
    classes = [elem for elem in list(pcd_pynt.points.columns) if elem not in removables]
    return classes
