
# Auto-WCEBleedGen Challenge


The Auto-WCEBleedGen challenge aims to provide a platform for the development, testing, and evaluation of Artificial Intelligence (AI) models for automatic detection and classification of bleeding and non-bleeding frames extracted from Wireless Capsule Endoscopy (WCE) videos. This project is the first of its kind, offering diverse training and test datasets while encouraging the development of vendor-independent, interpretable, and generalized AI models.
# Overview

Wireless Capsule Endoscopy (WCE) is a non-invasive medical imaging technology that allows for the examination of the gastrointestinal (GI) tract. The Auto-WCEBleedGen challenge focuses on automating the detection and classification of bleeding and non-bleeding frames within these videos.
# Datasets

### A) Training Datasets

The training dataset comprises 2618 annotated WCE frames collected from various internet resources. These frames cover a wide range of gastrointestinal bleeding cases throughout the GI tract.
Each frame is accompanied by medically validated binary masks and bounding boxes in three formats (txt, XML, and YOLO txt), providing precise annotations for model training.

### B)Test Dataset

The test dataset consists of independently collected WCE data, containing bleeding and non-bleeding frames from over 30 patients suffering from acute, chronic, and occult GI bleeding.
This dataset is sourced from patients referred at the Department of Gastroenterology and HNU, All India Institute of Medical Sciences, New Delhi, India.
# Objective

1)Develop AI models for the automatic detection and classification of bleeding and non-bleeding frames in WCE videos.

2)Promote the creation of vendor-independent models that can be applied across different WCE systems.

3)Encourage interpretability and generalization of AI models for this specific medical imaging application.
# Getting Started

To participate in the Auto-WCEBleedGen challenge, follow these steps:

1)Clone or download this repository to your local environment.

2)Access the provided training dataset for model development.

3)Utilize the independent test dataset for model evaluation.

4)Implement and train your AI models using the annotated frames and provided annotations.


# Implementaion by Convolutional Neural Network (CNN) 

A Convolutional Neural Network (CNN) is a type of neural network that is exceptionally effective for image classification tasks. It processes images in a way that preserves spatial relationships, which is crucial for understanding visual patterns.

## Model Architecture:
### 1)Input Layer:

This is where the model receives the input data. In this case, it expects images of a specific height, width, and color channels (3 for RGB images).

### 2)Convolutional Layers:
#### Conv2D(32, (3, 3), activation='relu'):

This is the first convolutional layer. It applies 32 filters (each with a 3x3 grid) to the input. The 'relu' activation function introduces non-linearity and helps the network learn complex patterns.
#### MaxPooling2D((2, 2)):

After each convolutional layer, we apply max-pooling, which reduces the spatial dimensions of the data. It effectively downsamples the features, focusing on the most important ones.

### 3)Second Convolutional Layer:

#### Conv2D(64, (3, 3), activation='relu'):
This is the second convolutional layer, which applies 64 filters.
S
### 4)Second MaxPooling Layer:

##### MaxPooling2D((2, 2)):
Similar to before, we apply max-pooling to downsample the features further.
### 5)Flatten Layer:

####Flatten():
This layer converts the multi-dimensional data into a one-dimensional array. It's a transition from convolutional layers to fully connected layers.
### 6)Fully Connected Layers:

#### Dense(64, activation='relu'):

This is a fully connected layer with 64 neurons, applying a 'relu' activation function. It helps in learning complex relationships in the data.
#### Dense(1, activation='sigmoid'):

This is the final layer. It has a single neuron with a 'sigmoid' activation function. For binary classification tasks (like ours), 'sigmoid' is commonly used as it squashes the output to be between 0 and 1, representing the probability of the class being positive (bleeding, in this case).
### 7)Compiling the Model:
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):

Here, we configure the model for training. We use the Adam optimizer, which is an efficient optimization algorithm. For a binary classification task, 'binary_crossentropy' is a suitable loss function. The metric we're interested in is accuracy.

## Evaluation of model

Accuracy: 0.9771

Precision: 0.9846

Recall: 0.9696

F1 score: 0.9770

ROX-AUC: 0.9944

Confusion Matrix:

[[257   4 ]

[ 8     255]]