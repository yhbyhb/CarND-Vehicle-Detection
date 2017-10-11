# **Vehicle Detection** 
by Hanbyul Yang, Oct 12, 2017

### Overview
This is a project of Self-Driving Car Nanodegree Program of Udacity.

The goals of this project is detecting vehicles of given image or videos that captured at driving car. 
Details of goals and steps are following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Additionally, apply a color transform and append binned color features, as well as histograms of color to HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

For the processing pipelines and codes, Check [`P5.ipynb`](./P5.ipynb).
I wrote this in the order given [rubrics](https://review.udacity.com/#!/rubrics/513/view).

[//]: # (Image References)
[cars]: ./output_files/cars.png "Cars"
[notcars]: ./output_files/notcars.png "Not cars"
[hog]: ./output_files/sample_hog_features.png "hog features"
[detection]: ./output_files/sample_detection.png
[scale_1]: ./output_files/sliding_window_scale_1.png
[scale_1.5]: ./output_files/sliding_window_scale_1.5.png
[scale_1.75]: ./output_files/sliding_window_scale_1.75.png
[scale_2]: ./output_files/sliding_window_scale_2.png
[heatmap]: ./output_files/test5_n_heatmap_nothres.png
[heatmap_thres]: ./output_files/test5_n_heatmap.png
[test1]: ./output_files/test1_output.png
[test2]: ./output_files/test2_output.png
[test3]: ./output_files/test3_output.png
[test4]: ./output_files/test4_output.png
[test5]: ./output_files/test5_output.png
[test6]: ./output_files/test6_output.png
[d_bb_hm_0]: ./output_files/difficult_bb_hm_0.png
[d_bb_hm_1]: ./output_files/difficult_bb_hm_1.png
[d_bb_hm_2]: ./output_files/difficult_bb_hm_2.png
[d_result_0]: ./output_files/difficult_result_0.png
[d_result_1]: ./output_files/difficult_result_1.png
[d_result_2]: ./output_files/difficult_result_2.png

### Writeup / README
This file `writeup.md` is for writeup. `README.md` describes contents (files and folders) briefly. 

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First of all, I leveraged the given codes of lessons which are in [`./helper_function.py`](./helper_function.py).

There are 8792 vehicle images and 8968 non-vehicle images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes. 3rd and 4th cell of jupyter notebook [`./P5.ipynb`](./P5.ipynb)

![alt text][cars]

![alt text][notcars]

Then, I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Also, I used binned color and histograms of color as features. The codes are located in function `extract_features()` in line 55 of [`./helper_function.py`](./helper_function.py)

I chose random images from each of the two classes and displayed them to get a feel for what the HOG features looks like. Below is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. For the convenience, I sticked with linear svm classifier and YCrCb color space.
    - Orientation 8 ~ 12.
    - Pixels per cell 8 ~ 16.
    - HOG Channels each one and all.
    - spatial_size 16, 32.
    - hist_bins 32, 64.

Final choice is in 2nd cell of jupyter notebook.
| Parameter | Final choice |
| ------------- |:-------------:|
| color_space | 'YCrCb' |
| orient | 9  |
| pix_per_cell | 8 |
| cell_per_block | 2 |
| hog_channel | "ALL" |
| spatial_size | (32, 32) |
| hist_bins | 32 |
| spatial_feat | True |
| hist_feat | True |
| hog_feat | True |
| y_start_stop | [400, 690] |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Feature normalization is performed and `sklearn.cross_validation.train_test_split()` is used for shuffling and train and test data split. 99.07% test accuracy is acquired. 5th cell of [jupyter notebook](./P5.ipynb) shows these process.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I also leveraged function `find_cars()` from lesson. It extracts hog features whole image at once rather than each window. The codes are located in 7th cell of notebook. `find_car()` uses 64 x 64 window size for 1.0 scale. Different scale extends or reduces region of sliding window by dividing image size with scale value.

At first I used two scales (1 and 2) but it performed poorly so I increase number of scales. Using 4 scales (1, 1.5, 1.75, 2) for searching vehicles was the optimal. 75% of overlapped window is used.
Here is an example. Each color represents different scale.

![alt text][detection]

But,there are false positives in image. So I made the heatmap for detected bounding boxes. High value means duplicated detections. 

![alt text][heatmap]

Thresholding heatmap with value 1 is used for removing false positive. The codes are in 11th and 12th cells of jupyter notebook.

![alt text][heatmap_thres]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tested with given test images in `test_images/` until it works all of test images. The optimization techniques I used are adjusting window size, using multiple windows sizes (scales) and adjusting heatmap threshold.

Belows are results of test images and 14th and 15th cell of jupyter notebook show the codes.

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
![alt text][test5]
![alt text][test6]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used two temporal (inter-frame) thresholding methods for remove false positives.
The first one is using all detected bounding boxes of recent 3 frames. Then applying heatmap thresholding with 3. It gives relieving temporal jitters.
The other one is thresholding change of centroid by comparing previous frame. 32 pixel is used for thresholding. It removes most of difficult false positives. 

Here are detected bounding box images of three frames and their corresponding heatmaps.

![alt text][d_bb_hm_0]
![alt text][d_bb_hm_1]
![alt text][d_bb_hm_2]

And then here are results after applying two temporal thresholding methods.

![alt text][d_result_0]
![alt text][d_result_1]
![alt text][d_result_2]

The code for processing pipeline is located in 13th cell of jupyter notebook.


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Even though I used very high accuracy (99.07%) of linear svm classifier and heat-map thresholding, the detected result was only good enough with good and clear image. If road is dirty which means color are not uniform, it usually has false positives. The most difficult case is when the shadows of trees are in the roads. 

My pipeline works poorly when car is on rightmost in the image. Because of the sliding window algorithm I used, there is some area not covered. here's the example.

![alt text][scale_1.75]
![alt text][scale_2]

I think more robust method could be one based on convolutional neural network, such as YOLO. It may have fewer false positives. 
