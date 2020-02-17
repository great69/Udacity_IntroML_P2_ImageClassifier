# Project 2: Image Classifier
This is the second project for Udacity's Intro to Machine Learning with TensorFlow Nanodegree Program to learn about Deep Learning using TensorFlow. The notebook utilizes [MobileNetV2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) feature extractor as the base model to classify novel flower images.

## Install
This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [TensorFlow](http://tensorflow.org)
- [TensorFlow Hub](https://www.tensorflow.org/hub)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

## Files
- `Project_Image_Classifier_Project.ipynb`: This is the main notebook to create and train a TensorFlow Keras model to predict the type of a flower from its image.
- `Project_Image_Classifier_Project.html`: HTML format of the notebook for viewing in browser
- `predict.py`: The command line application that implements the trained deep learning network to classify an input flower image.
- `model1581751950.h5`: The final deep learning model that predicts the flower type from its image with above 90% accuracy.
- `label_map.json`: The mapping of the key values to their corresponding flower types.
- `workspace-utils.py`: The supporting methods to keep the notebook active during long deep learning training phase.

## Run

In a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
ipython notebook Project_Image_Classifier_Project.ipynb
```  
or
```bash
jupyter notebook Project_Image_Classifier_Project.ipynb
```

This will open the iPython Notebook software and project file in your browser.

## Data
The dataset used is _[Oxford 102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)_.

## Notes
This project is for Udacity's Nanodegree Program and all the copyrights belong to Udacity.
