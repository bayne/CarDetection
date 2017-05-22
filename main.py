import zlib
import shelve
import glob
import numpy as np
import os.path
from classifier import CarClassifier

from moviepy.video.io.VideoFileClip import VideoFileClip

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
            feature_extractor_cache,
            region,
            min_window_size,
            max_window_size,
            window_x_step_ratio,
            window_y_step_ratio,
            scale_rate_y
    ) -> None:
        self.feature_extractor_cache = feature_extractor_cache
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
            feature_extractors[window_shape] = self.feature_extractor_cache.get(
                image=source_image,
                region=self.region,
                hog_color_space_convert=self.car_classifier.hog_color_space_convert,
                hog_channels=self.car_classifier.hog_channels,
                hog_pix_per_cell=(window_shape[0] // self.car_classifier.hog_pix_per_cell[0],
                                  window_shape[1] // self.car_classifier.hog_pix_per_cell[1]),
                hog_cell_per_block=self.car_classifier.hog_cell_per_block,
                hog_orient=self.car_classifier.hog_orient
            )

        rectangles = []
        for window in windows:
            try:
                features = feature_extractors[window['shape']].get(window, self.car_classifier.hog_pix_per_cell)

                sample = image[window['top_left']['y']:window['bottom_right']['y'], window['top_left']['x']:window['bottom_right']['x']]
                if self.car_classifier.classifier.predict(features)[0] == 1:
                    rectangles.append(window)
            except ValueError:
                pass

        return rectangles


class FeatureExtractorCache:
    def __init__(self, filename) -> None:
        self.filename = filename
        # self.cache = {}
        self.cache = shelve.open(filename)

    def save(self):
        if not isinstance(self.cache, dict):
            self.cache.close()

    def get(self, image=None, region=None, hog_color_space_convert=None, hog_channels=None,
            hog_cell_per_block=None, hog_orient=None, hog_pix_per_cell=None):
        # key = repr((image, region))
        key = str(zlib.crc32(image.data.tobytes())) + str(hog_pix_per_cell[0]) + str(hog_pix_per_cell[1])
        # key = zlib.crc32(bytes(repr([image_key, region['top_left']['x'], region['top_left']['y'], region['bottom_right']['x'],
        #             region['bottom_right']['y'], hog_color_space_convert, hog_channels, hog_cell_per_block, hog_orient,
        #             hog_pix_per_cell]), encoding='ascii'))
        feature_extractor = FeatureExtractor(
            region=region,
            hog_color_space_convert=hog_color_space_convert,
            hog_channels=hog_channels,
            hog_pix_per_cell=hog_pix_per_cell,
            hog_cell_per_block=hog_cell_per_block,
            hog_orient=hog_orient
        )

        if key in self.cache:
            feature_extractor.features = self.cache[key]
        else:
            feature_extractor.extract(image)
        self.cache[key] = feature_extractor.features
        return feature_extractor


class Heatmap:
    def __init__(self, image, cooldown_rate, warmup_rate, threshold) -> None:
        self.warmup_rate = warmup_rate
        self.threshold = threshold
        self.cooldown_rate = cooldown_rate
        self.__heatmap = np.zeros(np.shape(image))
        self.thresholded = np.zeros(np.shape(image))

    def add_rectangles(self, rectangles):
        self.__heatmap = np.array(np.subtract(self.__heatmap, self.cooldown_rate))
        self.__heatmap[self.__heatmap < 0] = 0
        self.__heatmap[self.__heatmap >= 1] = 1

        for rectangle in rectangles:
            self.__heatmap[
                rectangle['top_left']['y']:rectangle['bottom_right']['y'],
                rectangle['top_left']['x']:rectangle['bottom_right']['x']
            ] += self.warmup_rate

        self.thresholded = np.copy(self.__heatmap)
        self.thresholded[self.__heatmap < self.threshold] = 0


def get_pipeline():
    with open('classifier.p', 'rb') as file:
        car_classifier = pickle.load(file)

    feature_extractor_cache = FeatureExtractorCache('feature_extractor_cache.p')

    return Pipeline(
        car_classifier=car_classifier,
        feature_extractor_cache=feature_extractor_cache,
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
            'width': 72,
            'height': 72,
        },
        max_window_size={
            'width': 840,
            'height': 840
        },
        window_x_step_ratio=0.10,
        window_y_step_ratio=0.03,
        scale_rate_y=0.26
    )


def process_video():
    pipeline = get_pipeline()

    clip = VideoFileClip(filename="./project_video.mp4")

    class Processor:
        def __init__(self):
            self.heatmap = None

        def bounding_box(self, image):
            thresholded = np.multiply(np.copy(self.heatmap.thresholded), 255).round().astype('uint8')
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_RGB2GRAY)

            ret, thresh = cv2.threshold(thresholded, 0, 255, cv2.THRESH_BINARY)
            im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

            for contour in contours:
                M = cv2.moments(contour)
                x, y, w, h = cv2.boundingRect(contour)
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return image

        def process(self, image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.heatmap is None:
                self.heatmap = Heatmap(
                    image=image,
                    cooldown_rate=0.05,
                    warmup_rate=0.1,
                    threshold=0.8
                )
            pipeline.process(image, self.heatmap)
            image = self.bounding_box(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

    clip = clip.fl_image(Processor().process)

    clip.write_videofile(filename="./project_video_output.mp4", audio=False)
    pipeline.feature_extractor_cache.save()


def process_image():
    for file in glob.glob('test_images/test*.jpg'):
        image = cv2.imread(file)
        heatmap = Heatmap(
            image=image,
            cooldown_rate=0.2,
            warmup_rate=1.0,
            threshold=0.1
        )
        get_pipeline().process(image, heatmap)
        image = heatmap.annotate_heatmap(image)
        cv2.imwrite('output_images/' + file, image)

def test():
    pipeline = get_pipeline()
    for file in glob.glob('output_images/*.jpg'):
        image = cv2.imread(file)
        image = cv2.resize(image, (64,64))
        print(file, pipeline.car_classifier.classify(image))



def main():
    # process_image()
    # test()
    process_video()

if __name__ == "__main__":
    main()
