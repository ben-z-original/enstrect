from pathlib import Path
from enstrect.run import run
from enstrect.mapping.fuser import (
    NaiveMaxFuser, NaiveMeanFuser, NaiveMedianFuser,
    AngleBestFuser, AngleRangeFuser, AngleWeightedFuser,
    DistanceBestFuser, DistanceRangeFuser, DistanceWeightedFuser,
    AngleDistanceRangeBestFuser, AngleDistanceRangeFuser, NaiveAngleMaxRangeFuser)
from enstrect.segmentation.omnicrack30k_model import OmniCrack30kModel

if __name__ == "__main__":

    model = OmniCrack30kModel
    num = 4#20

    fusers = [NaiveMaxFuser, NaiveMeanFuser, NaiveMedianFuser,
    AngleBestFuser, AngleRangeFuser, AngleWeightedFuser,
    DistanceBestFuser, DistanceRangeFuser, DistanceWeightedFuser,
    AngleDistanceRangeBestFuser, AngleDistanceRangeFuser]

    for fuser in [NaiveAngleMaxRangeFuser]:
        for structure in [f"{key:05d}" for key in range(num)]:
            rootdir = Path(f"/home/chrisbe/repos/dissertation/enstrect/src/enstrect/assets/crackensembles/{structure}")
            obj_or_ply_path = rootdir / "mesh" / "mesh.obj"
            cameras_path = rootdir / "cameras.json"
            images_dir = rootdir / "views"

            out_dir = rootdir / "out" / f"{model.__name__}_{fuser.__name__}"
            (rootdir / f"out").mkdir(exist_ok=True)
            out_dir.mkdir(exist_ok=True)

            select_views = None  # ["0020", "0036", "0050"]#None #"0001"
            num_points = 50000  # 1000000
            scale = 1.0  # 0.25 # 1.0

            run(obj_or_ply_path, cameras_path, images_dir, out_dir, select_views, num_points, scale, fuser, model)
