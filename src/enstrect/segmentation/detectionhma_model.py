import cv2
import torch
import matplotlib
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from torchvision.transforms import Normalize
from detectionhma.inference import InferenceHMA
from torchvision.io import read_image, ImageReadMode
from enstrect.segmentation.base import SegmenterInterface

matplotlib.use("TkAgg")


class DetectionHMAModel(SegmenterInterface):
    def __init__(self, planpath=None, folds=(0,), allow_tqdm=True):


        self.infer = InferenceHMA(patch_size=1984, padding=32)

        self.infer.net.eval()
        #self.predictor.network.eval()

        self.classes = ["background", "crack", "spalling", "corrosion",
                        "efflorescence", "vegetation", "control_point"]

        self.class_weight = torch.tensor([1, 10, 4, 1, 1, 1, 1], dtype=torch.float16)

    def __call__(self, img: torch.Tensor):
        pass
        #img = img.to(torch.float32)
        #img = Normalize(img.mean((1, 2)), img.std((1, 2)))(img)
        #img = img.unsqueeze(1)
        #logits = self.predictor.predict_logits_from_preprocessed_data(img).squeeze()
        #softmax = torch.nn.functional.softmax(logits.to(torch.float32), dim=0)
        #return softmax


if __name__ == "__main__":

    model = DetectionHMAModel()
    example_img_path = Path(__file__).parents[1] / "assets" / "example_image.png"
    img = cv2.imread(str(example_img_path), cv2.IMREAD_COLOR)
    img = read_image(str(example_img_path), mode=ImageReadMode.RGB).to(torch.float32)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.to(torch.float32) / 255
    img = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = img.moveaxis(0, -1) #unsqueeze(1)
    #img = InferenceHMA.normalize(img)
    pred, attn = model.infer.run_large(img, attention=True)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(pred["pred"])
    plt.show()

    print()