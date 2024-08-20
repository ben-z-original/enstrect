import torch
import matplotlib
from pathlib import Path
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.io import read_image, ImageReadMode

matplotlib.use("TkAgg")


class SegmenterInterface(ABC):
    """
    A class to implement a unified interface for the image-level segmentation model.

    Methods
    -------
    __init__:
        Setting up the segmentation model.
    __call__:
        Run inference of the model on a rgb image, which is represented as a
        torch.Tensor of type torch.float32 and shape (3, height, width), i.e. channels-first mode).
        It returns a torch.Tensor of type torch.float32 and shape (num_classes, height, width).
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, img_rgb: torch.Tensor) -> torch.Tensor:
        pass


class RGBIdentityModel(SegmenterInterface):
    """
    A class for passing through the rgb image to illustrate the usage of the SegmenterInterface.
    """

    def __init__(self):
        self.model = torch.nn.Identity()

    def __call__(self, img_rgb):
        return self.model(img_rgb)


if __name__ == "__main__":
    identity_segmenter = RGBIdentityModel()
    example_img_path = (Path(__file__).parents[1] / "assets" / "example_image").with_suffix(".png")
    img_rgb_pyt = read_image(str(example_img_path), mode=ImageReadMode.RGB).to(torch.float32)
    probs = identity_segmenter(img_rgb_pyt)

    plt.subplot(121)
    plt.title("Input Image")
    plt.imshow(img_rgb_pyt.moveaxis(0, -1).to(torch.uint8))
    plt.subplot(122)
    plt.title("Output (same as input!)")
    plt.imshow(probs.moveaxis(0, -1).to(torch.uint8))
    plt.show()
