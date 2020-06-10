---
layout: page
title: Lane Detection
description: Finding Lane Lines on the Road
img: /assets/img/lane_detection/thumbnail.gif
---
<br />
I did this project as a part of the Udacity Self-Driving Car Nanodegree program. It is an implementation in Python 3 to detect lane lines on the road. In this blog post I have described the pipeline to achieve this task. The complete code for the project is available [here](https://github.com/sheelabhadra/CarND-LaneLines-P1).

### **Description of the pipeline**
---

My pipeline consists of 6 steps. For illustration I have used an example image shown below to show the effect of all the operations in my pipeline.

<p align="center">
  <img src="/assets/img/lane_detection/original_image.jpg">
</p>

#### 1. Color to grayscale conversion

The color image is converted to grayscale to make the computation easier and to detect lane lines of any color.

<p align="center">
  <img src="/assets/img/lane_detection/grayscale.jpg">
</p>

#### 2. Gaussian blur

Gaussian blur is applied on the grayscaled image to remove noise and spurious gradients. Gradients are calulated by the change in the intensity of gray value between adjacent pixels. More information about OpenCV's Gaussian blur can be found [here](http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur). I used a kernel size of 3 for Gaussian blur since the Canny edge detector (described next) already includes Gaussian smoothing.

<p align="center">
  <img src="/assets/img/lane_detection/Gaussian_blur.jpg">
</p>

#### 3. Canny transform

[Canny edge detection](http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html) is applied to the Gaussian-blurred image. This allows us to detect edges on the basis of change in the gradient intensity and direction at every pixel. The algorithm will first detect strong edge (strong gradient) pixels above the `high_threshold`, and reject pixels below the `low_threshold`. Next, pixels with values between the `low_threshold` and `high_threshold` will be included as long as they are connected to strong edges. The output edges is a binary image with white pixels tracing out the detected edges and black everywhere else. I used a `low_threshold` value of `50` and a `high_threshold` value of `150`.

<p align="center">
  <img src="/assets/img/lane_detection/Canny_edges.jpg">
</p>

#### 4. Region of interest mask

A mask is drawn to select only the relevant region in which there is a possibility of lane lines being present. This would be generally a quadrilateral shaped region in the bottom half of the image. The mask selects the white colors (edges) from the image obtained after applying Canny transformation using a bitwise and operation.

<p align="center">
  <img src="/assets/img/lane_detection/masked_image.jpg">
</p>

#### 5. Hough transform

[Hough transform](https://alyssaq.github.io/2014/understanding-hough-transform/) is applied on the the image obtained after applying the region of interest mask. The probabilisitc Hough transform is used to construct line segments using the edge points detected in the previous step. More information about using Hough transform can be found [here](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html). The edges are segments colored in red. The coordinates of the end points of the line segments are returned by the Hough transform function. 

<p align="center">
  <img src="/assets/img/lane_detection/line_edges.jpg">
</p>

#### 6. Drawing lane lines

The line segments obtained in the previous step are extrapolated to draw straight red colored lines along the lane lines. This is accomplished by modifying the `draw_lines()` function in the following ways:

* The slope of each line segment is calculated using the formula `(y2-y1)/(x2-x1)`.

* The slope of the lane lines lie within a particular range and this information is helpul in rejecting spurious line segments. I used a range of (0.5, 2.5) for the left lane line and (-2.5,-0.5) for the right lane line. Line segments with slope values outside of these range were ignored. The coordinates belonging to the left lane line were put into arrays `left_lane_x`, `left_lane_y` and the coordinates belonging to the right lane line were put into arrays `right_lane_x`, `right_lane_y`.

* The np.polyfit() function is helpful in constructing a line given a set of points. More information on this function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html). The degree was set to 1 and was applied to the arrays `left_lane_x`, `left_lane_y` to find the slope and intercept values of the left lane line and to the arrays `right_lane_x`, `right_lane_y` to find the slope and intercept values of the right lane line.

* The points of intersection of the 2 lane lines with the region of interest were calculated and the lane lines were drawn only inside the region of interest.

<p align="center">
  <img src="/assets/img/lane_detection/lane_lines.jpg">
</p>

#### Project Video
<br />
<p align="center">
	<iframe width="500" height="300" src="https://www.youtube.com/embed/q_gZGQMwX00" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>
<br />
### **Potential shortcomings**
---

* If the intensity of the lane lines varies a lot in the video due to different angles of the sun, then my pipeline does not perform well.
* If there are portions of the entire road that are white in color then the lane lines are not identified correctly.
* The lane lines are not stable at times when a few outlier points are detected.
* Due to the shortcoming mentioned above my pipeline performs awfully on the `challenge.mp4` video.
* My pipeline will not work on roads that do not contain lane markings 


### **Possible improvements**
---

* A possible improvement would be to track the yellow lane line by using color masking.
* The lane lines can be smoothed by keeping track of the lane lines in the previous video frames.