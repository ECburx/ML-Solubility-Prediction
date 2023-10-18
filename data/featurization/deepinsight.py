from typing import Union

from data.featurization.base import BaseRepresentation


class DeepInsight(BaseRepresentation):
    """
    Transforms features to an image matrix using dimensionality reduction and discritization.

    DeepInsight enables feature extraction through the application of CNN for non-image samples to seize imperative
    information. Features extrated from molecular descriptors are then transformed into images which can be fed to CNNs,
    increasing the versatility of CNN architectures.

    Sharma, A., Vans, E., Shigemizu, D., Boroevich, K.A. and Tsunoda, T. (2019). DeepInsight: A methodology to transform
    a non-image data to an image for convolution neural network architecture. Scientific Reports, [online] 9(1).
    doi:https://doi.org/10.1038/s41598-019-47765-6.
    """

    USER_WARNING_MSG = "[DeepInsight]: Xs must be converted to tabular data (unchecked)"

    def __init__(
            self,
            pixels: Union[int, tuple[int, int]],
            feature_extractor: str = "tsne",
            discretization: str = "bin",
    ):
        """
        :param pixels: int (square matrix) or tuple of ints (height, width) that defines the size of the image matrix.
        :param feature_extractor: string of value ('tsne', 'pca', 'kpca').
        :param discretization: string of values ('bin', 'assignment').
            Defines the method for discretizing dimensionally reduced data to pixel coordinates.
        """
        assert feature_extractor in ["tsne", "pca", "kpca"]
        self.feature_extractor = feature_extractor

        assert discretization in ["bin", "assignment"]
        self.discretization = discretization

        self.pixels = pixels

    def convert(self, Xs, ys=None):
        import warnings
        warnings.warn(DeepInsight.USER_WARNING_MSG, UserWarning)

        from pyDeepInsight import ImageTransformer
        from torchvision.transforms import Compose, ToTensor
        from pandas import DataFrame

        transformed_images = ImageTransformer(
            feature_extractor=self.feature_extractor,
            discretization=self.discretization,
            pixels=self.pixels
        ).fit_transform(Xs)
        composer = Compose([ToTensor()])

        return DataFrame({"image": [composer(img) for img in transformed_images]})
