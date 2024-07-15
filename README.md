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

### EDA
![image-2.png](attachment:image-2.png)

In the previous phase, we first analyzed the images in our dataset (i.e a subset of the Open Images Dataset V6) by various means, including computing the number of images, shape of the images dataframe, number of dog vs cat class images, size of the dataset and number of image aspect ratio types. We also changed the label names to human readable labels.

Upon analysis of the dataset, we found that it contains 12966 images in total (of dogs and cats), with a total size of 844.512 MB. The number of dog images outweighed the number of cat images, as there were 6855 dog images and 6111 cat images. We also plotted the different aspect ratios of images in the dataset as there were many different types of aspect ratios present in the dataset. We found that the most common aspect ratio was 512x384. We filtered image shapes with count less than 100 into a separate category called other to simplify the processing. 

We then plotted 6 random images using matplotlib along with their corresponding bounding boxes. We rescaled the images to 128x128 aspect ratio to standardize the images and save space/time for storing and processing.  

Before resizing: (notice how they are rectangular) <br>
![image-4.png](attachment:image-4.png)

After resizing to 128x128 aspect ratio: <br>
![image-5.png](attachment:image-5.png)

### Pytorch models for cat vs dog detection
Our methodology consists of three critical components:
#### MLP for Classification
A PyTorch-based MLP model designed to classify the content within the image, specifically distinguishing between categories such as 'cat' and 'dog'.
#### MLP for Bounding Box Regression
A parallel MLP model tasked with regressing the bounding box coordinates—[x, y, width, height]—of the detected object.
#### Multi-Headed Cat-Dog Detection
An innovative approach that integrates both classification and regression outputs through a multi-headed architecture, facilitating a combined learning process that leverages a multitask loss function composed of Cross-Entropy (CXE) for classification and Mean Squared Error (MSE) for bounding box regression.

Crucially, we incorporate the IoU metric, defined as IoU(A, B) = |A ∩ B| / |A ∪ B|, where A is the set of proposed object pixels, and B is the set of true object pixels. IoU serves as a robust measure for evaluating how well our proposed bounding boxes align with the ground truth.

![image.png](attachment:7ab8fb6f-5865-4f8e-a765-9b8d68948c18.png)


### Experiments
Project Workflow Block Diagram <br>
![image-3.png](attachment:image-3.png)

We first tried to undestand the problem statement, including what models we can create to do classification and localization. We then did data collection (simply used the pre-existing dataset). We then performed EDA to understand the structure of the data as well as some patterns. Then we did some data preprocessing, including normalization for the MLP models and changing size to fix amount. Then we created the models and ran them on the test set to compare them to one another, and upon finding the best model, we chose it to be our main model (for deployment in practical scenario). For MLP regression specifically, our main model was ran for more epochs to obtain a better score.


#### Metrics

##### Performance Metric: Accuracy

For the classifier machine learning pipelines, we evaluate performance primarily through the metric of accuracy. Accuracy is defined as the proportion of true results (both true positives and true negatives) among the total number of cases examined. Mathematically, it is expressed as:

$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$

where:
- $TP$ = True Positives
- $TN$ = True Negatives
- $FP$ = False Positives
- $FN$ = False Negatives

Accuracy provides a straightforward measure of how well our model correctly identifies or predicts the target variable.

##### SGD Classifier Pipeline: Utilizing MSE

For our Stochastic Gradient Descent Classifier (SGDC) pipeline, we employ the Mean Squared Error (MSE) as a specific performance metric. MSE measures the average squared difference between the estimated values and the actual value, offering insight into the precision of continuous value prediction within our classification process. MSE is given by:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

where:
- $n$ is the number of samples,
- $y_i$ is the actual value of the $i^{th}$ sample,
- $\hat{y}_i$ is the predicted value for the $i^{th}$ sample.

This formula allows us to quantify the deviation of the predicted continuous values from their actual values, thus offering a nuanced understanding of the model's predictive accuracy, especially in tasks where precision in the continuous output space is crucial.

##### Multi-Output Linear Regression Loss Function: Mean Squared Error (MSE)

The Mean Squared Error (MSE) is a common loss function used in regression, measuring the average squared difference between estimated values and the actual value. For a multi-output scenario, the MSE is calculated for each target independently and then averaged. The equation for MSE when extending to four targets is given by:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{1}{4} \sum_{j=1}^{4} (y_{ij} - \hat{y}_{ij})^2 \right) $$

where $n$ is the number of samples, $y_{ij}$ is the actual value for the $j^{th}$ target of the $i^{th}$ sample, and $\hat{y}_{ij}$ is the predicted value for the $j^{th}$ target of the $i^{th}$ sample.

##### Loss Function: Cross-Entropy + Mean Squared Error

To accommodate the dual objectives of classification and regression, we combine the Cross-Entropy (CXE) loss for the classification task with the Mean Squared Error (MSE) loss for the regression task. The total loss is a weighted sum of these two losses:

$$ Total Loss = \lambda \cdot CXE + (1 - \lambda) \cdot MSE $$

where $\lambda$ is a hyperparameter that balances the contribution of each loss component.

#### Results
##### Baseline 
Our experimental baseline models are as follows: <br>
![image-11.png](attachment:image-11.png)

NOTE: we used the same family of input features for both, i.e all of the features in the dataset.

##### Pytorch models 
Images are first normalized using z-score normalization to ensure that the pixel values have a mean of 0 and a standard deviation of 1. <br>
![image.png](attachment:673f6dec-0e52-4d83-a20e-833c0b5fa711.png)

Then the dataset is split into training, validation, and testing parts.  1% of the data is kept for testing, and 10% of the training data is used as a validation set.<br>
![image.png](attachment:91158c57-daa0-48c3-ad75-dc07b5e799a2.png)

Different hyperparameters are tried out in our MLP model for image classification to see which gives the best results.  Hyperparameters include the dimension of the hidden layers and the learning rate.  Adam optimizer and CrossEntropyLoss are used.
We first tried 5 different combinations of hidden layers and learning rate, and found the following: <br>
![image-8.png](attachment:image-8.png)

The best performing MLP model has 256 hidden dimensions and a learning rate of 0.001 (with 2 hidden layers and ReLU activation function after each linear transformation in hidden layers), and it got a test accuracy of 53.8%, which is worse than our baseline Linear Model.

It seems as we increase the hidden layer dimensions, the test accuracy increases. So we then test with 3 more models with higher hidden layer dimensions, and found the following: <br>
![image-9.png](attachment:image-9.png)

The best MLP classification model came out to be MLP Model 8, with hidden layer dimensions = 2042, a learning rate of 0.001 and an accuracy of 60%. It exceeds the baseline accuracy of the linear SGD classification model.

We then test different hidden layer dimension and learning rates for the MLP regression model as well, this time with number of epochs = 100 at first and using MSE as the loss function along with Adam optimization.
We then choose the best MLP regression model (model 5) and run it for 1000 epochs to obtain our best model. <br>
![image.png](attachment:image.png)

For multiheaded dog/cat detection, we made a model that consists of two separate multi-layer perceptrons (MLPs): one for classification and another for regression. Each MLP has three fully connected layers followed by ReLU activation functions. The classification MLP takes an input tensor and outputs a classification result with a size defined by output_dim_cls, representing the number of classes. Meanwhile, the regression MLP takes the same input tensor and produces a regression output with a size defined by output_dim_reg, indicating the number of regression outputs. The model's architecture enables it to handle both classification and regression tasks simultaneously. 

We ran the model for 30 epochs for 3 different hidden_dim and learning rate combinations. In this case, we chose to only increase the hidden_dim values as it seemed learning rate didn't make a big difference in our previous models. We plotted out the graphs with epochs as the x axis and metric values as the y axis (MSE/CXE). We had a batch size of 500 so we loaded 500 images every epoch to mitigate memory allocation issues and to speed up the training process. We generated two graphs: one that shows the zoomed out MSE and CXE loss function for every epoch during training, and a zoomed in version (as it starts off with a loss much greater than 1). The graphs are as follows: <br>
![image-6.png](attachment:image-6.png)

We then find the mAP (IOU) results for each, and plot the results in a dataframe:<br>

![image-7.png](attachment:image-7.png)

The best performing model for cat/dog detection is Model 0, with hidden_dim=64 and lr=0.001, which got an mAP Score of 0.4306. This model will act as the baseline pipeline for our next phase. We have every model from the experiments saved as .pt files, which can be used as pre-trained model at any time.

In total, we ran 8 experiments for MLP classification, 6 experiments for MLP regression, and 3 experiments for the multiheaded model. 

#### EfficientDet models
When testing two different EfficientDet models, both performed poorly.
![image-10.png](attachment:image-10.png)
Notably, this implementation experiences large memory requirements and high latency due to very large time and space complexities. We therefore limited our approach to only two epochs, possibly contributing to the poor results.

#### Fully Convolutional Neural Network
Our FCN model achieved a superior test accuracy to our other models with 70.9% accuracy.
![image-12.png](attachment:image-12.png)

Our bounding box predictions seemed to be biased towards dogs, as shown below.
![output.png](attachment:output.png)


### Discussion
In phase 3, we investigated the performance of MLP models for image classification, regression, and multiheaded dog/cat detection. Overall, the results were promising for image classification and multiheaded detection, but less conclusive for regression.

For image classification, we found that increasing the hidden layer dimension of the MLP significantly improved performance. The best MLP model achieved an accuracy of 60%, exceeding the baseline accuracy of the linear SGD classifier (57.7%). This suggests that MLPs can be effective for this task, particularly with careful hyperparameter tuning.

For MLP regression, the findings were less impressive. The best MLP regressor model achieved a test MSE loss of 0.55, which is way higher compared to the baseline linear regression model (0.035). It's possible that exploring a wider range of hyperparameters or introducing more complex model architectures might be necessary for regression tasks.

The multiheaded model for dog/cat detection achieved a best mAP score of 0.4306. While this is a moderate score, it demonstrates the potential of this approach for combining classification and regression tasks within a single model. Further exploration with different hyperparameter settings or model architectures could potentially improve these results. It should be noted that the MSE score for the best multiheaded model was still lower than the baseline linear regression model's MSE score, but a lot closer than the MLP regression model. The multiheaded model achieved an MSE score of 0.04 on the test set, while the baseline linear regression model achieved an MSE score of 0.035.

We were able to improve our multiheaded model mAP score by elevating it to 0.77, however, the bounding box predictions for our multi-headed model are biased towards predicting dogs. Additionally, the coordinates of the bounding box seem localized to the bottom right, both indicating possible rescaling issues when normalizing the training set.

Overall, our FCN achieved superior accuracy to our baseline and our pytorch models, indicating the predictive prowess of full connected CNNs over our multiheaded approach. However, our multheaded approach is still superior in regression tasks.

### Conclusion
Our project focuses on the development of robust models for cat and dog detection (CaDoD), aiming to accurately classify images containing these animals and precisely localize their bounding boxes. 

Our hypothesis posited that machine learning pipelines with custom features can accomplish the task of cat/dog detection and classification. We conducted EDA, implemented baseline models using sklearn/pytorch, and evaluated their performance using relevant metrics. Our experiments with MLP classifiers and regressors revealed promising results, with the best MLP classification achieving a 60% accuracy, closely trailing our baseline SGDClassifier. However, the regression model fell short, indicating room for improvement in localizing bounding boxes. Our multiheaded model showcased the potential of combining classification and regression tasks, achieving an mAP score of 0.4306.

While certain configurations show promise, others indicate a need for further optimization. The insights gained from this phase provide a foundation for subsequent iterations. Moving forward, we envision incorporating deep learning models such as CNNs, EfficientNets and SWIN transformers to improve our mAP score.

### Bibliography
None
