# Computer Vision Interview Questions and Answers

Welcome to the **Computer Vision Interview Questions and Answers** repository! This collection aims to provide fundamental and advanced computer vision questions along with their solutions to help you prepare for technical interviews or improve your understanding of computer vision concepts.

### Table of Contents

1. [Introduction](#introduction)
2. [Questions and Answers](#questions-and-answers)
3. [Acknowledgments](#acknowledgments)

...
## Questions-and-Answers
## 1. What is Computer Vision and How Does it Relate to Human Vision?

**Computer Vision** is a field of Artificial Intelligence that enables computers to understand and interpret visual information from images and videos. It is inspired by the way humans see and process visual information, aiming to make machines perform similar visual tasks such as recognizing objects, faces, and scenes.

While human vision relies on the brain to process visual inputs, computer vision uses algorithms and models to analyze and understand visual data. The goal is to replicate human-like visual perception in machines.

## Human and Computer Vision: A Comparison

### Comparison Table

|   | Human Vision | Computer Vision |
|---|--------------|-----------------|
| **Data Input**   | Sensed by eyes, then registered and processed by the brain | Captured through cameras and video devices |
| **Hardware** | Eyes, optic nerves, and the visual cortex | Cameras, storage devices, and processors (such as CPUs or GPUs) |
| **Perception** | Real-time visual comprehension with recognized patterns, depth, and motion | Data-driven analysis to identify objects, classify scenes, and extract features |
| **Object Recognition** | Contextual understanding with the ability to recognize familiar or unfamiliar objects based on prior knowledge | Recognition based on statistical models, trained on vast amounts of labeled data |
| **Robustness** | Adapts to varying environmental conditions, such as lighting changes and occlusions | Performance affected by factors like lighting, image quality, and occlusions |
| **Educative Process** | Gradual learning and refinement of vision-related skills from infancy to adult stages | Continuous learning through exposure to diverse visual datasets and feedback loops |

### Detailed Insights

#### Human Vision

- **Retina Details**: The human retina measures about 5 × 5 cm and contains approximately $10^8$ sampling elements (rods and cones).
- **Spatial Resolution**: About 0.01° over a 150° field of view, not evenly spaced. The fovea has high resolution, while the peripheral region has lower resolution.
- **Intensity Resolution**: Approximately 11 bits per element, with a spectral range of 400–700 nm.
- **Temporal Resolution**: Around 100 ms (10 Hz).
- **Data Rate**:  😲 Two eyes together have a data rate of about 3 GBytes per second!
- **Importance of Vision**: Vision is the most powerful of our senses. About 1/3 of our brain is dedicated to processing visual signals.
- **Visual Cortex**: Contains around 10^11 neurons, showcasing the complexity of human vision processing.

#### Vision as Data Reduction

- **Raw Data Feed**: From camera/eyes: approximately $10^7$ to $10^9$ Bytes per second.
- **Edge and Feature Extraction**: Data reduced to around $10^3$ to $10^4$ Bytes per second after processing edges and salient features.
- **High-Level Scene Interpretation**: Further reduced to about $10$ to $10^2$ Bytes per second.



## 2. How to rotate an image by an arbitrary angle ?

Rotating an image involves transforming the coordinates of each pixel using a rotation matrix. This mathematical operation allows us to reposition pixels and achieve a rotated version of the image.
### Mathematical Explanation

#### Rotation Matrix and Point Transformation

To rotate a 2D point by an angle $\theta$, the rotation matrix is given by:

$$
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
$$

This matrix, when multiplied with a vector, rotates the vector counterclockwise by the angle $\theta$.

#### Translation to the Origin

To rotate a point around a specific point $(cx, cy)$ rather than the origin $(0, 0)$, we first need to translate the point $(x, y)$ such that $(cx, cy)$ becomes the origin. This is done by subtracting the center coordinates:

$$
\begin{bmatrix}
x - cx \\
y - cy
\end{bmatrix}
$$

This vector represents the relative position of the point $(x, y)$ with respect to the center $(cx, cy)$.

#### Applying the Rotation Matrix

After translating the point to be centered around the origin, we apply the rotation matrix to the translated point:

$$
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
\cdot
\begin{bmatrix}
x - cx \\
y - cy
\end{bmatrix}
$$

This multiplication rotates the translated point around the new origin (which is now at $(0, 0)$).

#### Translating Back to the Rotated Image Center

After the rotation, we translate the coordinates back to the new center $(cx', cy')$ of the rotated image. This step is necessary because the new image might have different dimensions (especially if the rotation is not by 90, 180, or 270 degrees):

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
\cdot
\begin{bmatrix}
x - cx \\
y - cy
\end{bmatrix}
+
\begin{bmatrix}
cx' \\
cy'
\end{bmatrix}
$$

This final translation ensures the rotated point $(x', y')$ is correctly positioned in the new image space.

The new coordinates are calculated relative to the center of the image to ensure the rotation occurs around the image center.

### Steps to Rotate an Image

1. **Convert the Angle to Radians**: Since trigonometric functions in math libraries typically use radians, convert the angle from degrees to radians.
2. **Compute the Rotated Image Dimensions**: The dimensions of the rotated image will generally differ from the original image. Calculate the new dimensions to ensure the rotated image fits fully.
3. **Create an Empty Array for the Rotated Image**: Initialize an array to store the pixel values of the rotated image.
4. **Apply Rotation Matrix for Each Pixel**: For each pixel in the output image, compute its original position using the inverse of the rotation matrix. If it maps to a valid position in the original image, copy the pixel value.
5. **Adjust Pixel Mapping**: Ensure that the pixel mapping is within the bounds of the original image to avoid errors.

### Optimized Python Implementation

Below is an optimized Python implementation that rotates an image by a specified angle without using external libraries like `cv2` for rotation.

```python
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    angle_rad = math.radians(angle) # convert image to radians 
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    h, w = image.shape[:2]          # Dim of input image
    cx, cy = w // 2, h // 2   # original image center 
    new_w = int(abs(w * cos_theta) + abs(h * sin_theta))
    new_h = int(abs(w * sin_theta) + abs(h * cos_theta))
    rotated_image = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    cx', cy' = new_w // 2, new_h // 2  # new image center 

    for i in range(new_h):
        for j in range(new_w):
            x = (j - cx') * cos_theta + (i - cy') * sin_theta + cx
            y = -(j - cx') * sin_theta + (i - cy') * cos_theta + cy
            if 0 <= x < w and 0 <= y < h:
                rotated_image[i, j] = image[int(y), int(x)]

    return rotated_image

if __name__ == "__main__":
    # Load an image
    image = cv2.imread("ena.png")  # Use your image path here
    image_np = np.array(image)
    # Rotate the image by 45 degrees
    rotated_image_np = rotate_image(image_np, 45)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(rotated_image_np, cv2.COLOR_BGR2RGB)) # CV2 (BGR) , matplotlib(RGB)
    plt.axis('off')  # Turn off axis labels
    plt.show()

```

## 3. How does _edge detection_ work in image analysis?

**Edge detection** methods aim to find the boundaries in images. This step is vital in various computer vision tasks, such as object recognition, where edge pixels help define shape and texture. An edge is a location where we have a rapid change of image intensity. 

### Types of Edges in Images

Modern edge detection methods are sensitive to various types of image edges:

- **Step Edges**: Rapid intensity changes in the image.
- **Ramp Edges**: Gradual intensity transitions.
- **Roof Edges**: Unidirectional edges associated with a uniform image region.

### Sobel Edge Detection Algorithm

The Sobel operator is one of the most popular edge detection methods. It calculates the gradient of the image intensity by convolving the image with small[ square `3x3` convolution kernels][4]. One for detecting the x-gradient and the other for the y-gradient.

These kernels are:

#### $G_x$

$$
G_x = 
\begin{bmatrix}
+1 & 0 & -1 \\
+2 & 0 & -2 \\
+1 & 0 & -1
\end{bmatrix}
$$

#### $G_y$

$$
G_y = 
\begin{bmatrix}
+1 & +2 & +1 \\
0 & 0 & 0 \\
-1 & -2 & -1
\end{bmatrix}
$$

The magnitude $G$ and direction $\theta$ of the gradient are then calculated as:

$$
G = \sqrt{G_x^2 + G_y^2}
$$

$$
\theta = \arctan\left(\frac{G_y}{G_x}\right)
$$

The calculated $G$ and $\theta$ are used to detect edges.

### Canny Edge Detection

The Canny edge detector is a multi-step algorithm which can be outlined as follows:

1. **Noise Reduction**: Apply a Gaussian filter to smooth out the image.
2. **Gradient Calculation**: Use the Sobel operator to find the intensity gradients.
3. **Non-Maximum Suppression**: Thins down the edges to one-pixel width to ensure the detection of only the most distinct edges.
4. **Double Thresholding**: To identify "weak" and "strong" edges, pixels are categorized based on their gradient values.
5. **Edge Tracking by Hysteresis**: This step defines the final set of edges by analyzing pixel gradient strengths and connectivity.

### Implementations in Python

Here is the Python code:

#### Using Canny Edge Detection from OpenCV

```python
import cv2

# Load the image in grayscale
img = cv2.imread('image.jpg', 0)

# Apply Canny edge detector
edges = cv2.Canny(img, 100, 200)

# Display the original and edge-detected images
cv2.imshow('Original Image', img)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### Using Sobel Operator from OpenCV

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv2.imread('image.jpg', 0)

# Compute both G_x and G_y using the Sobel operator
G_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
G_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Compute the gradient magnitude and direction
magnitude, direction = cv2.cartToPolar(G_x, G_y)

# Display the results
plt.subplot(121), plt.imshow(magnitude, cmap='gray')
plt.title('Gradient Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(direction, cmap='gray')
plt.title('Gradient Direction'), plt.xticks([]), plt.yticks([])
plt.show()
```
<br>

## 4. What is Non-Maximum Suppression (NMS) and Intersection over Union (IoU)?

### Non-Maximum Suppression (NMS)

Non-Maximum Suppression (NMS) is a technique used in object detection to eliminate redundant bounding boxes around the same object. When an object detection model identifies an object, it often generates multiple bounding boxes with different confidence scores. NMS helps in retaining only the most accurate bounding box by suppressing the others.

#### How NMS Works:

1. **Score Thresholding:** Filter out all bounding boxes with confidence scores below a predefined threshold.
2. **Sort by Confidence:** Sort the remaining bounding boxes by their confidence scores in descending order.
3. **Select the Highest Score Box:** Select the bounding box with the highest confidence score.
4. **Compute IoU:** Compute the Intersection over Union (IoU) of the selected box with the remaining boxes.
5. **Remove Overlapping Boxes:** Remove boxes that have an IoU above a certain threshold with the selected box.
6. **Repeat:** Repeat the process until no more boxes remain.

NMS ensures that the final bounding boxes are accurate and do not overlap significantly, retaining only the best bounding box for each object.

### Intersection over Union (IoU)

Intersection over Union (IoU) is a metric used to measure the overlap between two bounding boxes, which is crucial for evaluating the accuracy of object detection models.

$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} $

- **Area of Overlap:** The area where the predicted bounding box overlaps with the ground truth bounding box.
- **Area of Union:** The total area covered by both the predicted and ground truth bounding boxes.

#### IoU Values:

- **0:** No overlap between the two bounding boxes.
- **1:** Perfect overlap between the predicted and ground truth bounding boxes.
- A typical IoU threshold (e.g., 0.5) is used to determine if a detection is a true positive (IoU >= threshold) or a false positive (IoU < threshold).

### Summary

- **Non-Maximum Suppression (NMS):** A technique used in object detection to remove redundant bounding boxes and retain the most accurate ones.
- **Intersection over Union (IoU):** A metric to measure the overlap between two bounding boxes, helping to evaluate the performance of object detection models. [source](https://www.youtube.com/watch?v=mS_csnzZJ-o)
















...

## Acknowledgments

This repository leverages knowledge from various open-source resources, including textbooks, online courses, and research papers. Special thanks to the contributors and the open-source community for their invaluable resources.
