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















...

## Acknowledgments

This repository leverages knowledge from various open-source resources, including textbooks, online courses, and research papers. Special thanks to the contributors and the open-source community for their invaluable resources.
