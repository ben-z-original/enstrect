import numpy as np
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN


def prepare_exposed_rebar(pcd_pynt, eps_m=0.01):
    pcd_pynt.points["exposed_rebar"] = pcd_pynt.points["spalling"] + pcd_pynt.points["corrosion"]
    idxs = np.where(pcd_pynt.points["exposed_rebar"] == 1)[0]

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
            #pcd_pynt.points.loc[idxs_cluster, "corrosion"] = 0  # keep corrosion as is

    return pcd_pynt


def get_pcd_classes(pcd_pynt):
    removables = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "argmax", "cluster"]
    classes = [elem for elem in list(pcd_pynt.points.columns) if elem not in removables]
    return classes
