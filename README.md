# Multi-Task Object Detection and Localization for Cats and Dogs (CaDoD)

## Report

### Abstract

Our project involves the development of robust models for cat and dog detection. We aim to enhance the accuracy and efficiency of identifying cats and dogs within images using MLP and multiheaded models. In this phase, we aimed to create MLP models for classification and regression, and a multiheaded model to do both. We first analyzed the images in our dataset by various means, including computing number of dog vs cat class images and plotting random images with bounding boxes. We then rescaled the images and created a baseline SGDClassifier model for classifying dog vs cat and a Linear Regression model for bounding box labelling. We first tested out several hidden dimensions and learning rates for the MLP classification/regression and multiheaded models, and settled for the model with the best accuracy/MSE and mAP score. We found that our MLP classification model was able to obtain an accuracy of 60%, a ~2% difference from our baseline SGDClassifier model. Our best MLP regression model obtained an MSE of 0.55 on our test set, which is worse than our linear regression model (0.03 MSE). Our best multiheaded model obtained an mAP score of 0.4306.

### Introduction
In our project, we take on the challenging task of classifying cat vs dog pictures and create bounding boxes around them. Our goal is to develop robust models capable of accurately identifying and localizing cats and dogs in images. The purpose of this project is create an end to end pipeline in machine learning to create an object detector for cats and dogs. There are about 13,000 images of varying shapes and aspect ratios. They are all RGB images and have bounding box coordinates stored in a .csv file. In order to create a detector, we will first have to preprocess the images to be all of the same shapes, take their RGB intensity values and flatten them from a 3D array to 2D. Then we will feed this array into a linear classifier and a linear regressor to predict labels and bounding boxes.

Further more, utilizing the versatile and powerful PyTorch framework, we develop MLP models optimized for these tasks in phase 3. Our experiments are designed to refine these models to improve their performance in classifying images into categories and accurately predicting the bounding box coordinates of objects within those images. The ultimate goal is to build a reliable model that performs well against established benchmarks, paving the way for practical applications in various fields, including automated surveillance, image retrieval, and autonomous vehicles.

### Dataset
The dataset we've opted for is a subset of the Open Images Dataset V6, a comprehensive collection of labeled images spanning various categories.  It contains millions of labeled images spanning a wide variety of categories including cat and dog images, making it a valuable resource for training and evaluating object detection machine learning models. Our subset specifically focuses on images featuring cats and dogs as the primary subjects, with accompanying bounding box annotations indicating the precise location of each animal within the image. Our subset will have 12,866 images of dogs and cats. Each image is supplemented with metadata such as dimensions, file paths, and class labels, facilitating efficient model training and evaluation. The vastness of the dataset ensures a rich diversity of images capturing different breeds, poses, and environments, providing ample variation for robust model training.

We will be using a subset of the Open Images Dataset V6. It is a large-scale dataset curated by Google designed to facilitate computer vision research and development. It contains millions of labeled images spanning a wide variety of categories, making it a valuable resource for training and evaluating machine learning models. Our subset will have 12,866 images of dogs and cats.

The image archive `cadod.tar.gz` is a subset [Open Images V6](https://storage.googleapis.com/openimages/web/download.html). It contains a total of 12,966 images of dogs and cats.


Image bounding boxes are stored in the csv file `cadod.csv`. The following describes whats contained inside the csv.

* ImageID: the image this box lives in.
* Source: indicates how the box was made:
    * xclick are manually drawn boxes using the method presented in [1], were the annotators click on the four extreme points of the object. In V6 we release the actual 4 extreme points for all xclick boxes in train (13M), see below.
    * activemil are boxes produced using an enhanced version of the method [2]. These are human verified to be accurate at IoU>0.7.
* LabelName: the MID of the object class this box belongs to.
* Confidence: a dummy value, always 1.
* XMin, XMax, YMin, YMax: coordinates of the box, in normalized image coordinates. XMin is in [0,1], where 0 is the leftmost pixel, and 1 is the rightmost pixel in the image. Y coordinates go from the top pixel (0) to the bottom pixel (1).
* XClick1X, XClick2X, XClick3X, XClick4X, XClick1Y, XClick2Y, XClick3Y, XClick4Y: normalized image coordinates (as XMin, etc.) of the four extreme points of the object that produced the box using [1] in the case of xclick boxes. Dummy values of -1 in the case of activemil boxes.

The attributes have the following definitions:

* IsOccluded: Indicates that the object is occluded by another object in the image.
* IsTruncated: Indicates that the object extends beyond the boundary of the image.
* IsGroupOf: Indicates that the box spans a group of objects (e.g., a bed of flowers or a crowd of people). We asked annotators to use this tag for cases with more than 5 instances which are heavily occluding each other and are physically touching.
* IsDepiction: Indicates that the object is a depiction (e.g., a cartoon or drawing of the object, not a real physical instance).
* IsInside: Indicates a picture taken from the inside of the object (e.g., a car interior or inside of a building).
For each of them, value 1 indicates present, 0 not present, and -1 unknown.

### Pipelines

#### Stochastic Gradient Descent Classifier (SGDC) as Baseline Classifier

Before delving into our custom implementations, it's important to establish the Stochastic Gradient Descent Classifier (SGDC) as our baseline model. SGDC, a linear regressor widely used for classification tasks, offers a robust point of comparison for our more specialized models. Its efficiency and simplicity make SGDC an excellent linear regression baseline for large data, against which the performance and complexity of our homegrown models can be evaluated.

#### Multi-Output Linear Regression as Baseline Regressor

Linear Regression is a fundamental algorithm in machine learning for predicting continuous values. In our custom model, we extend this concept to predict multiple outputs at once, targeting scenarios where we need to predict four related values, such as the coordinates of a bounding box in an image (x, y, width, height). This extension aims to improve upon the baseline by offering predictions that are not just categorical but spatially informative as well.

#### MLP Classifier/Regressor

Multilayer Perceptron models are algorithms in ML that are able to do both classification and regression. MLPs can capture complex non-linear relationships in the data, which is often essential for distinguishing between different classes in image data. In our MLP classification, we have 49152 input features (pixels for each image), 2 hidden layers with varying number of neurons (tested multiple models), and 1 2-neuron output layer. We use RelU activation between each layer. For our MLP regression model, we have 49152 input features, 2 hidden layers with varying number of neurons (tested multiple models), and 1 4-neuron output layer (i.e the coordinates for each of the 4 corners of the bounding boxes).

#### Multiheaded Model

Our multi headed model is a neural network architecture designed for the combined tasks of image classification and object localization, specifically for distinguishing between images of dogs and cats and predicting bounding boxes around the detected objects. Our multiheaded model uses CXE as the classification loss metric and MSE as the regression loss metric. The classification branch comprises three fully connected layers (fc_cls1, fc_cls2, and fc_cls3). The first layer takes an input of 49,152 features, likely representing the flattened output of a feature extractor or a convolutional neural network (CNN). This layer reduces the features to 512 dimensions, which is further processed by the subsequent layers to make a final prediction. The final layer of this branch outputs a probability distribution over two classes: dog and cat.

On the other hand, the regression branch handles the bounding box prediction task. It also consists of three fully connected layers (fc_reg1, fc_reg2, and fc_reg3). Like the classification branch, the first layer takes the same 49,152-dimensional input and reduces it to 512 dimensions. The following layers further refine this representation to produce a final output consisting of four values. These four values represent the coordinates of the bounding box: (x_min, y_min, x_max, y_max).


#### EfficientDet
We use https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch pre-trained EfficientDet models, which we fine-tune for our multitask purpose of cats vs dogs classification and localization. This implementation uses tensorboard to visualize results. We run the python py files in terminal (as we were having issues with running it on jupyter), and we paste the per epoch results as comments.

EfficientDet enhances object detection by integrating the EfficientNet backbone with a Feature Pyramid Network (FPN) and a Bi-directional Feature Pyramid Network (BiFPN). The EfficientNet serves as the base for feature extraction, balancing model size and accuracy. FPN and BiFPN refine the feature maps across different scales, facilitating the detection of objects of various sizes. Multiple detectors, such as EfficientDet-D0 to EfficientDet-D7, are built on top of these feature maps to predict bounding boxes, object classes, and scores.

For training, EfficientDet uses a combination of loss functions: Regression Loss (CXE in the implementation) penalizes the difference between predicted and ground-truth bounding box coordinates, Classification Loss (MSE in the implementation) computes the probability of the anchor box containing an object for each class using cross-entropy loss, and Focal Loss further addresses class imbalance by down-weighting the loss for well-classified examples. The model is trained to minimize the weighted sum of these losses using gradient descent optimization, aiming to optimize both accuracy and efficiency in object detection. 

EfficientDet models, ranging from EfficientDet-D0 to EfficientDet-D7, vary in terms of their depth, width, and resolution, offering a trade-off between speed and accuracy. EfficientDet-D0 is the smallest and fastest variant, suitable for applications requiring real-time inference on resource-constrained devices. In contrast, EfficientDet-D7 is the largest and most accurate variant, providing higher precision at the cost of increased computational resources and inference time. As you move from D0 to D7, the model's depth, width, and resolution increase, enabling it to capture more complex features and detect objects with greater accuracy, but requiring more computational power for inference.
