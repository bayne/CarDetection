import argparse

from skimage.feature import hog
import cv2
import numpy as np


class FeatureExtractor:
    def __init__(
            self,
            image=None,
            region=None,
            hog_color_space_convert=False,
            hog_channels=(0, 1, 2),
            hog_pix_per_cell=(8, 8),
            hog_cell_per_block=(2, 2),
            hog_orient=9,
    ) -> None:
        super().__init__()
        self.region = region
        self.image = image
        self.hog_orient = hog_orient
        self.hog_cell_per_block = hog_cell_per_block
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_color_space_convert = hog_color_space_convert
        self.hog_channels = hog_channels
        self.features = []

    def extract(self):
        # Create a list to append feature vectors to
        self.features = []

        if self.hog_color_space_convert:
            feature_image = cv2.cvtColor(self.image, self.hog_color_space_convert)
        else:
            feature_image = np.copy(self.image)

        for channel in self.hog_channels:

            feature = hog(
                # image=feature_image[:, :, channel],
                image=feature_image[self.region['top_left']['y']:self.region['bottom_right']['y'], self.region['top_left']['x']:self.region['bottom_right']['x'], channel],
                orientations=self.hog_orient,
                pixels_per_cell=self.hog_pix_per_cell,
                cells_per_block=self.hog_cell_per_block,
                feature_vector=False,
                block_norm='L1'
            )
            feature_image_shape = np.shape(feature_image)
            zero_shape = (feature_image_shape[0]//self.hog_pix_per_cell[0] - 1, feature_image_shape[1]//self.hog_pix_per_cell[1] - 1, self.hog_cell_per_block[0], self.hog_cell_per_block[1], self.hog_orient)
            zero_feature = np.zeros(zero_shape)
            region_offset = {
                'top_left': {
                    'x': self.region['top_left']['x'] // self.hog_pix_per_cell[0],
                    'y': self.region['top_left']['y'] // self.hog_pix_per_cell[1]
                },
                'bottom_right': {
                    'x': self.region['bottom_right']['x'] // self.hog_pix_per_cell[0],
                    'y': self.region['bottom_right']['y'] // self.hog_pix_per_cell[1]
                },

            }

            feature_shape = np.shape(feature)

            zero_feature[
                region_offset['top_left']['y']:region_offset['top_left']['y']+feature_shape[0],
                region_offset['top_left']['x']:region_offset['top_left']['x']+feature_shape[1]
            ] = feature

            feature = zero_feature

            self.features.append(feature)

        return np.ravel(self.features)

    def get(self, window, classifier_hog_pix_per_cell):
        top_left = (
            window['top_left']['x'] // self.hog_pix_per_cell[0],
            window['top_left']['y'] // self.hog_pix_per_cell[1]
        )
        bottom_right = (
            window['bottom_right']['x'] // self.hog_pix_per_cell[0],
            window['bottom_right']['y'] // self.hog_pix_per_cell[1]
        )
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]

        top_left = (
            top_left[0] - (classifier_hog_pix_per_cell[0] - 1 - width),
            top_left[1] - (classifier_hog_pix_per_cell[1] - 1 - height)
        )

        features = [
            features_channel[
                top_left[1]:bottom_right[1],
                top_left[0]:bottom_right[0]
            ]
            for features_channel in self.features
        ]
        return np.ravel(features)


def main(args):
    feature_extractor = FeatureExtractor(
        image=cv2.imread(args.image_filename),
        hog_color_space_convert=cv2.COLOR_RGB2HSV,
        hog_channels=(1, 2),
        hog_pix_per_cell=(8, 8),
        hog_cell_per_block=(2, 2),
        hog_orient=9
    )

    feature_extractor.extract()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument('image_filename', type=str, help='The directory of PNG files for vehicles')

    args = parser.parse_args()
    main(args)
