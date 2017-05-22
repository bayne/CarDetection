# Vehicle Detection and Tracking

https://github.com/udacity/CarND-Vehicle-Detection

> The goals / steps of this project are the following:
> 
> * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
> * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
> * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
> * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
> * Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
> * Estimate a bounding box for vehicles detected.

![screen shot 2017-05-21 at 7 00 41 pm](https://cloud.githubusercontent.com/assets/712014/26289834/e6d58c1a-3e57-11e7-9166-b20932b7b0c1.png)

## Feature extraction

To detects cars in the image we need to be able to classify subsamples of the full frame into two categories: car and non-car. A classifier requires us to provide it features which it will use to determine if the image is a car or not. In this project I focused on using a particular type of feature called a Histogram of Oriented Gradients.

### Histogram of Oriented Gradients (HOG)

> The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image.
[Wikipedia](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)

I wrapped the call to the `skimage.feature.hog()` function in my own class called [`FeatureExtractor`](https://github.com/bayne/CarDetection/blob/a942c3cb87224ce3763939874620c4210edaa828/feature_extractor.py#L8-L8). I intended for this class to encapsulate the logic for extracting features from a given image.  It was also useful for optimization in later parts of the pipeline.

#### Parameter selection

I choose the HOG parameters through trial and error and manually modifying them until I got better results. The quality of the results were evaluated based on timeliness and accuracy of the final output. The `pixels_per_cell` parameter however varied based on the sliding window scale so that the features from the video would match the feature size from the training data set.

## Training the classifier

The classifier was trained using only HOG features from the saturation and value channels of the HSV image. I initially only used the saturation channel but later found that including the value channel improved the performance significantly. I used a support vector machine to power the classifier via `sklearn.svm.LinearSVC()`.

I wrapped the classifier in my own class called [`CarClassifier`](https://github.com/bayne/CarDetection/blob/a942c3cb87224ce3763939874620c4210edaa828/classifier.py#L13-L13). The purpose for this was to split out the training step out from the rest of the code and limit the pickling to only the `CarClassifier`.

## Sub-sampling the full frame

Since the goal of the project is to find cars in a given frame, we need some type of mechanism to get subsamples of the full frame. The classifier is fed these subsamples and tells us which is a car and which is not a car. The approach I went with for generating these subsamples was a sliding window search.

**Frame**
![test6](https://cloud.githubusercontent.com/assets/712014/26292006/7af901d8-3e67-11e7-9d49-3bf0e447cc8c.jpg)

**Car**
![601199](https://cloud.githubusercontent.com/assets/712014/26291971/3ef6bdec-3e67-11e7-8b78-70a8b809e946.jpg)

**Not a car**
![7269436](https://cloud.githubusercontent.com/assets/712014/26291979/4f35317a-3e67-11e7-8c68-bbb272c4815c.jpg)

### Sliding window search

**Intermediate step (pipeline demonstration)**
![image](https://cloud.githubusercontent.com/assets/712014/26292355/1af57e08-3e6a-11e7-8f7e-eb54b38afd84.png)

The simplest approach would be just to slide a window across the entire image of varying sizes. Although this would result in the highest number of possible true positives, it would also be too computationally expensive. In this project one of the constraints is to reduce the amount of time required to process each frame.

[My implementation](https://github.com/bayne/CarDetection/blob/a942c3cb87224ce3763939874620c4210edaa828/main.py#L62-L62) of the sliding window search generates several windows with the given attributes:

- Smaller windows are closer to the middle of the frame (farther away cars are smaller)
- The top half of the frame is excluded from being searched

I used a couple of tunable parameters to find the best distribution of windows:

#### `region`

A rectangle the defines where to generate the windows in, this is how I prevented the window search from searching in the sky.

#### `window_y_step_ratio`

Defined the amount of overlap each window had when it slid in the `y` direction

#### `window_x_step_ratio`

Defined the amount of overlap each window had when it slid in the `x` direction

#### `scale_rate_y`

The rate at which the windows grew as they approached the bottom of the frame

#### `min_window_size`

The smallest window size which is found nearest the top of the region

## Optimization

The HOG feature extractor is an expensive operation so optimization efforts were focused on reducing the number of calls to that function. Initially I implemented the feature extractor to extract on demand when given a subsample. Using this method resulted in the same areas of the image having the operation performed on it more than once. I optimized this process by:

1. Get every window size that will be used
1. [Create a `FeatureExtractor` for each given window size](https://github.com/bayne/CarDetection/blob/a942c3cb87224ce3763939874620c4210edaa828/main.py#L112)
1. Extract the HOG features for each given window size in the region of interest
2. [Subsample the HOG features for any given window](https://github.com/bayne/CarDetection/blob/a942c3cb87224ce3763939874620c4210edaa828/feature_extractor.py#L51)

### Development optimization

Since my development cycle involved a significant amount of parameter tuning, it payed off to reduce the amount of time required to generate an output video. The generation of the HOG features was also saved to disk to be used for future iterations. This was implemented in a class I refer to as the [`FeatureExtractorCache`](https://github.com/bayne/CarDetection/blob/a942c3cb87224ce3763939874620c4210edaa828/main.py#L138)

## Processing the video

[video](https://drive.google.com/file/d/0B1CQ1n9EZIF6RjBOOFhYSW1DeDg/view?usp=sharing)

![[video](https://drive.google.com/file/d/0B1CQ1n9EZIF6RjBOOFhYSW1DeDg/view?usp=sharing)](https://media.giphy.com/media/3ohzdKBQXhY9JOw29a/giphy.gif)

Up to this point the pipeline was focused on a single frame. When given a video, there is additional information that can be used to accurately find a car in the image. I utilized a heatmap that persisted from frame to frame and was the underlying structure that powered the eventual annotations (bounding boxes around the cars) on the frame.

Once again I turned the heatmap into a class which I called [`Heatmap`](https://github.com/bayne/CarDetection/blob/a942c3cb87224ce3763939874620c4210edaa828/main.py#L172-L172). The heatmap collected the windows that were identified as cars by the classifier and attempted to filter out false positives.

![image](https://cloud.githubusercontent.com/assets/712014/26293022/86e9c3e4-3e6f-11e7-86d3-3b94b8af2cf8.png)

The filtering of the false positives were controlled by the following tunable parameters:

**`warmup_rate`**

This is the value that gets added to the heatmap when a given pixel is found in a window that was positively identified as a car

**`cooldown_rate`**

The rate at which all pixels in the heatmap decrease. This removed the false positives from persisting in the heatmap

**`threshold`**

The minimum value of a pixel in the heatmap to be considered as a true positive. This is eventually used when generating the bounding box.

### Bounding box

The bounding box was the final output and what was used to annotate the frame with a true positive of a car. This was relatively straightforward and merely used a contour finding function provided by OpenCV. The [contour was generated](https://github.com/bayne/CarDetection/blob/a942c3cb87224ce3763939874620c4210edaa828/main.py#L237) around the thresholded heatmap values.

## Possible problems/improvements

- Try a different classifier
  - The current classifier produces a fair amount of false positives
- Look at a different approach for the contour finding
  - If cars have overlapping heatmaps they will result in the same contour
