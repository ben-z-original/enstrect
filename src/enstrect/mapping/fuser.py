import torch


class Fuser:
    """
    Class for implementing the logic of fusing the probabilities of the image-level segmentation.
    """

    def __init__(self, class_weight=None):
        self.class_weight = class_weight

    def __call__(self, probabilities, viewing_conditions):
        probabilities = probabilities * self.class_weight if self.class_weight is not None else probabilities

        # apply angle constraint on views
        probabilities[(230 < viewing_conditions.angles) + (viewing_conditions.angles < 130), :] = torch.nan

        aggr = torch.nanmean(probabilities, axis=1)

        # Note: aggr[:, 0] is assumed to be the default/background class, adjust accordingly
        aggr[:, 1:] = torch.nan_to_num(aggr[:, 1:], nan=0.0)
        aggr[:, 0] = torch.where(torch.isnan(aggr[:, 0]), 1 - aggr[:, 1:].sum(dim=1), aggr[:, 0])
        argmax = torch.argmax(aggr, dim=1)
        return aggr, argmax
