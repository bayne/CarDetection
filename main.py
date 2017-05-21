import copy
import glob
import numpy as np

from moviepy.video.io.VideoFileClip import VideoFileClip

from classifier import CarClassifier
from feature_extractor import FeatureExtractor
import pickle
import cv2
import os


class ImageSaver:
    def __init__(self, output_directory, enabled) -> None:
        """
        :param output_directory: The directory to write the sub directories for the output images
        :param enabled: Set to true to enable writing of the images
        """
        self.enabled = enabled
        self.__output_directory = output_directory

    def save(self, sub_directory, filename, image):
        """
        Saves the image to the disk
        
        :param sub_directory: The subdirectory for the image to fall under
        :param filename: The filename of the image
        :param image: The image data
        :return: 
        """
        if self.enabled:
            directory = self.__output_directory + '/' + sub_directory + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(filename=directory + '/' + filename, img=image)


class Pipeline:
    def __init__(
            self,
            car_classifier,
            region,
            min_window_size,
            max_window_size,
            window_x_step_ratio,
            window_y_step_ratio,
            scale_rate_y
    ) -> None:
        self.window_y_step_ratio = window_y_step_ratio
        self.window_x_step_ratio = window_x_step_ratio
        self.scale_rate_y = scale_rate_y
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.region = region
        self.car_classifier = car_classifier

    def get_windows(self):
        """
        Return a set of windows in x and y
        :param y: 
        :return: 
        """
        vertical_offset = 0
        windows = []
        window_height = self.min_window_size['height']
        window_width = self.min_window_size['width']
        while vertical_offset + (self.region['top_left']['y'] + window_height) < self.region['bottom_right']['y']:
            horizontal_offset = 0

            while horizontal_offset + (self.region['top_left']['x'] + window_width) < self.region['bottom_right']['x']:
                horizontal_offset += int(self.window_x_step_ratio * window_width)

                window = {
                    'top_left': {
                        'x': self.region['top_left']['x'] + horizontal_offset,
                        'y': self.region['top_left']['y'] + vertical_offset
                    },
                    'bottom_right': {
                        'x': self.region['top_left']['x'] + horizontal_offset + window_width,
                        'y': self.region['top_left']['y'] + vertical_offset + window_height
                    },
                    'shape': (window_width, window_height)
                }

                windows.append(window)

            vertical_offset += int(self.window_y_step_ratio * window_height)
            window_width += int(window_width * self.scale_rate_y)
            window_height += int(window_height * self.scale_rate_y)

        return windows

    def process(self, image, heatmap):
        rectangles = self.__get_rectangles(image)
        heatmap.add_rectangles(rectangles=rectangles)

    def __get_rectangles(self, image):
        """
        Annotates an image with bounding boxes around cars
        
        :param image: 
        """
        windows = self.get_windows()
        source_image = image.copy()
        window_shapes = set([window['shape'] for window in windows])
        feature_extractors = {}
        for window_shape in window_shapes:
            feature_extractors[window_shape] = FeatureExtractor(
                image=source_image,
                region=self.region,
                hog_color_space_convert=self.car_classifier.hog_color_space_convert,
                hog_channels=self.car_classifier.hog_channels,
                hog_pix_per_cell=(window_shape[0] // self.car_classifier.hog_pix_per_cell[0],
                                  window_shape[1] // self.car_classifier.hog_pix_per_cell[1]),
                hog_cell_per_block=self.car_classifier.hog_cell_per_block,
                hog_orient=self.car_classifier.hog_orient
            )
            feature_extractors[window_shape].extract()

        rectangles = []
        for window in windows:
            try:
                features = feature_extractors[window['shape']].get(window, self.car_classifier.hog_pix_per_cell)

                if self.car_classifier.classifier.predict(features)[0] == 1:
                    rectangles.append(window)
            except ValueError:
                print(np.shape(features))
                pass

        return rectangles

class Heatmap:

    def __init__(self, image) -> None:
        self.__heatmap = np.zeros_like(image)

    def add_rectangles(self, rectangles):
        for rectangle in rectangles:
            self.__heatmap[
                rectangle['top_left']['y']:rectangle['bottom_right']['y'],
                rectangle['top_left']['x']:rectangle['bottom_right']['x']
            ] += 1

    def annotate_heatmap(self, image):
        return cv2.addWeighted(image, 1.0, self.__heatmap, 25.0, 1.0)

def get_pipeline():
    with open('classifier.p', 'rb') as file:
        car_classifier = pickle.load(file)

    return Pipeline(
        car_classifier=car_classifier,
        region={
            'top_left': {
                'x': 0,
                'y': 400
            },
            'bottom_right': {
                'x': 1240,
                'y': 600
            }
        },
        min_window_size={
            'width': 64,
            'height': 64,
        },
        max_window_size={
            'width': 840,
            'height': 840
        },
        window_x_step_ratio=0.1,
        window_y_step_ratio=0.01,
        scale_rate_y=0.2
    )


def process_video():
    pipeline = get_pipeline()

    clip = VideoFileClip(filename="./project_video.mp4")

    pipeline.current_filename = "0_frame.png"

    def process(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heatmap = Heatmap(image)
        pipeline.process(image, heatmap)
        image = heatmap.annotate_heatmap(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    clip = clip.fl_image(process)
    clip.write_videofile(filename="./project_video_output.mp4", audio=False)


def process_image():
    for file in glob.glob('test_images/*'):
        image = cv2.imread(file)
        heatmap = Heatmap(image)
        get_pipeline().process(image, heatmap)
        image = heatmap.annotate_heatmap(image)
        cv2.imwrite('output_images/'+file, image)


def main():
    # process_image()
    process_video()


if __name__ == "__main__":
    main()
