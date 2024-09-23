import cv2
import torch
import gdown
import zipfile
import matplotlib
import numpy as np
import gradio as gr
from pathlib import Path
from argparse import ArgumentParser
from skimage.morphology import thin
from matplotlib import pyplot as plt
from torchvision.transforms import Normalize
from enstrect.segmentation.base import SegmenterInterface
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

matplotlib.use("TkAgg")

class OmniCrack30kModel(SegmenterInterface):
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
            planpath = Path(__file__).parent / "checkpoints" / "OmniCrack30kModel_nnUNetTrainer__nnUNetPlans__2d"

        # doẃnload and unzip plan
        zippath = Path(planpath).with_suffix(".zip")
        if not zippath.exists() and not Path(planpath).exists():
            url = "https://drive.google.com/uc?id=15S1dvjr7050kISlQ0JTiEPA1eeUDfoOl"
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
        self.preprocessor = self.predictor.configuration_manager.preprocessor_class()

        self.classes = ["background", "crack"]

        self.class_weight = torch.tensor([1, 10], dtype=torch.float16)

    def __call__(self, img: torch.Tensor):
        img = img.to(torch.float32)
        img = Normalize(img.mean((1, 2)), img.std((1, 2)))(img)
        img = img.unsqueeze(1)
        logits = self.predictor.predict_logits_from_preprocessed_data(img).squeeze()
        softmax = torch.nn.functional.softmax(logits.to(torch.float32), dim=0)
        return softmax