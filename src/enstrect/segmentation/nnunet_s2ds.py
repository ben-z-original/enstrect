import torch
import gdown
import zipfile
import matplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from torchvision.transforms import Normalize
from torchvision.io import read_image, ImageReadMode
from enstrect.segmentation.base import SegmenterInterface
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class NNUNetS2DSModel(SegmenterInterface):
    def __init__(self, planpath=None, folds=(0,), allow_tqdm=True):
        # instantiate the nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=allow_tqdm,
        )

        if planpath is None:
            planpath = Path(__file__).parent / "checkpoints" / "nnUNetTrainer__nnUNetPlans__2d"

        # doẃnload and unzip plan
        zippath = Path(planpath).with_suffix(".zip")
        if not zippath.exists() and not Path(planpath).exists():
            url = "https://drive.google.com/uc?id=1UeXzpH76GYtZtyn2IjhDvD5Qu3u91YcC"
            Path(planpath.parent).mkdir(exist_ok=True)
            gdown.download(url, str(zippath), quiet=False)

        if not Path(planpath).exists():
            with zipfile.ZipFile(str(zippath), 'r') as zip_ref:
                zip_ref.extractall(str(planpath))

        # initializes the network architecture, loads the checkpoint
        self.predictor.initialize_from_trained_model_folder(
            str(planpath),
            use_folds=folds,
            checkpoint_name='checkpoint_final.pth',
        )

        self.predictor.network.eval()

        self.classes = ["background", "crack", "spalling", "corrosion",
                        "efflorescence", "vegetation", "control_point"]

        self.class_weight = torch.tensor([1, 10, 4, 1, 1, 1, 1], dtype=torch.float16)

    def __call__(self, img: torch.Tensor):
        img = img.to(torch.float32)
        img = Normalize(img.mean((1, 2)), img.std((1, 2)))(img)
        img = img.unsqueeze(1)
        logits = self.predictor.predict_logits_from_preprocessed_data(img).squeeze()
        softmax = torch.nn.functional.softmax(logits.to(torch.float32), dim=0)
        return softmax


if __name__ == "__main__":
    segmenter = NNUNetS2DSModel()

    example_img_path = Path(__file__).parents[1] / "assets" / "example_image.png"
    img_rgb_pyt = read_image(str(example_img_path), mode=ImageReadMode.RGB).to(torch.float32)

    probs = segmenter(img_rgb_pyt)
    argmax = torch.argmax(probs, dim=0)

    matplotlib.use("TkAgg")  # required since nnunet uses non-gui backend
    plt.subplot(121)
    plt.title("Input Image")
    plt.imshow(img_rgb_pyt.moveaxis(0, -1).to(torch.uint8))
    plt.subplot(122)
    plt.title("Output")
    plt.imshow(argmax)
    plt.show()
