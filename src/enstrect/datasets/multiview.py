import torch
from pathlib import Path
from tensordict import TensorDict
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from enstrect.datasets.utils import json_to_pyt3d


class MultiviewDataset(Dataset):
    def __init__(self, cameras_pyt3d_or_path, views_dir=None, select_views=None, scale=1.0, device="cuda:0"):
        self.scale = scale
        self.device = device
        self.views_dir = Path(views_dir) if views_dir is not None else None

        if isinstance(cameras_pyt3d_or_path, (Path, str)):
            self.view_keys, self.cameras = json_to_pyt3d(cameras_pyt3d_or_path,
                                                         select_views=select_views,
                                                         scale=scale, device=device)
        else:
            self.cameras = cameras_pyt3d_or_path
            self.view_keys = [f"{key:04d}" for key in range(len(self.cameras))]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx, rgb=False):
        camera = self.cameras[idx]

        if self.views_dir is not None:
            img = read_image(str(next(self.views_dir.glob(f"{self.view_keys[idx]}*")))).to(torch.float32)
            img = img[[2, 1, 0], :, :] if not rgb else img
        else:
            img = None

        transforms = v2.Compose([
            v2.Resize(camera.image_size[0].to(torch.int).tolist()),
        ])
        img = transforms(img)

        out = {"name": self.view_keys[idx],
               "camera": camera,
               "image": img}
        return TensorDict(out)
