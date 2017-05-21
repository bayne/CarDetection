import argparse
import pickle
from feature_extractor import FeatureExtractor
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import glob
import cv2
import numpy as np


class CarClassifier:
    def __init__(
            self,
            hog_color_space_convert=False,
            hog_channels=(0, 1, 2),
            hog_pix_per_cell=(8, 8),
            hog_cell_per_block=(2, 2),
            hog_orient=9,
            classifier=None
    ) -> None:
        super().__init__()
        self.hog_orient = hog_orient
        self.hog_cell_per_block = hog_cell_per_block
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_color_space_convert = hog_color_space_convert
        self.hog_channels = hog_channels
        self.classifier = classifier
        self.score = None
        self.fit_time = None

    def extract_features(self, image):
        # Create a list to append feature vectors to
        feature_extractor = FeatureExtractor(
            image=image,
            hog_color_space_convert=self.hog_color_space_convert,
            hog_channels=self.hog_channels,
            hog_pix_per_cell=self.hog_pix_per_cell,
            hog_cell_per_block=self.hog_cell_per_block,
            hog_orient=self.hog_orient
        )

        return feature_extractor.extract()

    def train(self, car_filenames, not_car_filenames):

        car_features = []
        notcar_features = []

        for filename in car_filenames:
            image = cv2.imread(filename=filename)
            image = cv2.resize(image, (64, 64))
            car_features.append(self.extract_features(image))

        for filename in not_car_filenames:
            image = cv2.imread(filename=filename)
            image = cv2.resize(image, (64, 64))
            notcar_features.append(self.extract_features(image))

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        rand_state = np.random.randint(0, 100)

        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        start = time.time()
        self.classifier.fit(X_train, y_train)
        end = time.time()
        self.fit_time = end - start

        self.score = self.classifier.score(X_test, y_test)

    def classify(self, image):
        features = self.extract_features(image)
        return self.classifier.predict(features)


def main(args):
    car_classifier = CarClassifier(
        hog_color_space_convert=cv2.COLOR_RGB2HSV,
        hog_channels=(1, 2),
        hog_pix_per_cell=(8, 8),
        hog_cell_per_block=(2, 2),
        hog_orient=9,
        classifier=LinearSVC()
    )

    car_classifier.train(
        car_filenames=glob.iglob(args.vehicles_dir+'/**/*.png'),
        not_car_filenames=glob.iglob(args.non_vehicles_dir+'/**/*.png')
    )
    print(car_classifier.fit_time, 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(car_classifier.score, 4))
    with open(args.classifier_output_file, 'wb') as file:
        pickle.dump(car_classifier, file)
        print('Classifier saved to: ', args.classifier_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the classifier')
    parser.add_argument('vehicles_dir', type=str, help='The directory of PNG files for vehicles')
    parser.add_argument('non_vehicles_dir', help='The directory of PNG files for non-vehicles')
    parser.add_argument('classifier_output_file', help='The name of the file to output the classifier to')

    args = parser.parse_args()
    main(args)
