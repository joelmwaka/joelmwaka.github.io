---
layout: post
title:  "Automatic Object Detection Dataset Creation with Python"
date:   2023-10-29 10:00:00 +0000
categories: Dataset Classification OpenCV 
---

The success of object detection models, such as YOLO (You Only Look Once) and Faster R-CNN (Region-based Convolutional Neural Network), heavily relies on the quality of the datasets used for training. These models are at the forefront of computer vision applications, enabling tasks like real-time object identification, tracking, and localization. The importance of high-quality datasets in their training can't be overstated, as they serve as the foundation upon which these models learn to understand and recognize objects within images or video streams.

In the journey to create these datasets, a unique set of challenges emerges. Ensuring the datasets are vast, diverse, and meticulously annotated with object classes and bounding boxes can be a painstaking endeavor. It's a labor-intensive process, often fraught with human error. Recognizing this, and inspired by the increasing demand for high-quality datasets, I embarked on a mission to simplify and automate this crucial step in the development of object detection models.

This article introduces a Python-powered solution that can be used by developers and researchers to effortlessly generate datasets tailored to object detection tasks limited to 2D planar objects.

## Project Overview

**Objective**: This project aims to automate data annotation, generate diverse data, incorporate data augmentation, and ensure data quality for object detection datasets. 

**Limitation**: The data set generation algorithm is currently limited to only 2D planar objects like road signs, book covers, playing cards, license plates, etc.

**Requirements**: We use Python 3.8 with Numpy and OpenCV.

## Methodology

To demonstrate the application of this tool, some popular roads signs will be used. We shall try to create a dataset with right-of-way signs in germany. These signs are designed to demonstrate who has right of way (priority) at a junction. The following signs are all in use in germany:

<p align="center">
  <img src="https://github.com/joelmwaka/joelmwaka.github.io/blob/main/_posts/images/2023-10-2023_right-of-way-signs.PNG?raw=true">
</p>

I assume that the object images are tightly boung around the objects that you want to detect - this means we can therefore assume that the shape and the size of the image describes the bounding box of the object in the image. In case you wish to select features in the images, this tool offers that option too. 

### Step 1: Annotate the original object images.

The bounding box of will automatically be extracted from the shape and size of the object object image you provide. This tool however also provides the option to annotate features in the objects.

The object images should besaved as a name depicting the label of the objectc. For example, the stop sign object image should be named 'stop.png'. All these object images are then saved in the folder './data/objects/images/'. The tool will then save the labels of each image and get the the dimensions of the input images. The dimensions are assumed to be the bounding box of the object in the image provided.

The stored labels and bounding boxes are saved in JSON files under the folder './data/objects/annotations/' and look as follows:

***object_labels.json***

```json
{
  "0": {
    "object_image": "0.png", 
    "label": "end-priority-road"
  }, 
  "1": {
    "object_image": "1.png", 
    "label": "priority-over-oncoming-traffic"
  }, 
  "2": {
    "object_image": "2.png", 
    "label": "priority"
  }, 
  "3": {
    "object_image": "3.png",
    "label": "priorty-road"
  }, 
  "4": {
    "object_image": "4.png", 
    "label": "stop"}, 
  "5": {
    "object_image": "5.png", 
    "label": "yield-uncoming-traffic"}, 
  "6": {
    "object_image": "6.png", 
    "label": "yield"
  }
}
```

***boundboxes.json***
```json
{
  "0": {
    "top_left": [0, 0], 
    "bottom_right": [120, 120]
  }, 
  "1": {
    "top_left": [0, 0], 
    "bottom_right": [120, 120]
  }, 
  "2": {
    "top_left": [0, 0], 
    "bottom_right": [120, 105]
  }, 
  "3": {
    "top_left": [0, 0], 
    "bottom_right": [120, 120]
  }, 
  "4": {
    "top_left": [0, 0], 
    "bottom_right": [120, 120]
  }, 
  "5": {
    "top_left": [0, 0], 
    "bottom_right": [120, 120]
  },
  "6": {
    "top_left": [0, 0], 
    "bottom_right": [120, 105]
  }
}
```
In case you wish to annotate some feature points in the images as well, you can run the main script with the flag features. This will open up a window with the image that will allow you to then pick features from the image. Just follow the instructions given to you in terminal. The features are also stored in a JSON file named 'features.json' in the same folder as the two JSON files above. For each of the image, you can specify the number of features you want to annotate and then click on the features in the image to select them.

Now that we have our labels, original bounding boxes and features, we can look at the next step which is the actual data set generation.

### Step 2: Generate synthetic data set.

This step is split into the following sub steps:

1. **Background image acquisition**. In this step, we acquire an image that will serve as the background of our object. To do this, we have two options. We can either provide a path to a folder full of random background PNG images or provide a path to a video from which we shall acquire random frames and use these as the background of our object. To select either one of these options, run the code with the argument --background_source <path to either videos (MP4) or images (PNG)>.
2. **Object image alteration**. In this sub step, the orientation of the original image is altered. We apply some homographic operations to the image. Some noise and minor occlusions are added to the image. We also get the new bounding box and features (if features were picked) of the object in the image after applying the homographic transformation.
3. **Paste altered object in a random background image**. We can now paste the object image we created in the previous step in a random background image. Thereby, we shall acquire and save the bounding box of the object and the new positions of the features in the new generated image. This bounding box and features are stored in two seperate JSON files in the folder ./data/dataset/annotations/raw_annotations/'.
4. **Augment final image**. In the last sub step of step 2, we manipulate the image by performing a sequence of image alteration functions like adding noise, altering the brightness, adding artificial shadows, among others. The final image is the stored in the folder './data/dataset/images/'.

### Step 3: (Optional) Transform raw annotations to COCO format.


## Results visualizer

After the code is done running and generating a dataset for your deep learning purpose, you can use the same code with the flag '--visualize' to look through final images generated with their respective annotations.

## Final remarks

The code for this project is currently still under development and will be pushed soon. Thanks!

## References

1. Ref 1.