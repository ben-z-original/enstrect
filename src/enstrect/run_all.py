from pathlib import Path
from enstrect.run import run
from enstrect.mapping.fuser import NaiveMeanFuser, AngleBestFuser, NaiveMaxFuser
from enstrect.segmentation.omnicrack30k_model import OmniCrack30kModel

if __name__ == "__main__":

    fuser = NaiveMaxFuser
    model = OmniCrack30kModel

    for structure in ["indoors", "beam", "wall", "bridge_b", "bridge_s"]:
        for segment in ["segment2"]:
            rootdir = Path(
                f"/home/chrisbe/repos/dissertation/enstrect/src/enstrect/assets/crackstructures/{structure}/{segment}")
            obj_or_ply_path = rootdir / "mesh" / "mesh.obj"
            cameras_path = rootdir / "cameras.json"
            images_dir = rootdir / "views"
            out_dir = rootdir / f"out_{model.__name__}_{fuser.__name__}"
            select_views = ["0006", "0008"] #None #["0020", "0036", "0050"]#None #"0001"
            num_points = 1000000
            scale = 1.0 #0.25 # 1.0 # 0.25 # 1.0

            run(obj_or_ply_path, cameras_path, images_dir, out_dir, select_views, num_points, scale, fuser, model)

    for structure in ["wall", "beam", "bridge_b", "bridge_s", "indoors"]:
        for segment in ["segment1", "segment3"]:
            rootdir = Path(
                f"/home/chrisbe/repos/dissertation/enstrect/src/enstrect/assets/crackstructures/{structure}/{segment}")
            obj_or_ply_path = rootdir / "mesh" / "mesh.obj"
            cameras_path = rootdir / "cameras.json"
            images_dir = rootdir / "views"
            out_dir = rootdir / f"out_{model.__name__}_{fuser.__name__}"
            select_views = None #["0020", "0036", "0050"]#None #"0001"
            num_points = 1000000
            scale = 1.0 # 0.25 # 1.0

            run(obj_or_ply_path, cameras_path, images_dir, out_dir, select_views, num_points, scale, fuser, model)
