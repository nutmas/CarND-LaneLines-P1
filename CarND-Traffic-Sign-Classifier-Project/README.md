## Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

This project utilises neural networks to classify road traffic signs. The [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) it used to train and validate a model which is then tested on some other traffic sample signs obtained from the internet.

This repository contains my model which can read in road traffic images and produce a classification for that image. It also contains the pipeline used to train and validate the model.

---
The goals / steps of this project are the following:

* Load the data set
* Explore, summarise and visualise the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyse the softmax probabilities of the new images
* Summarise the results with a written report

---

###Check list:

* Trained Model to classify road signs: [Final_Trained_sign_classifier_model.data-00000-of-00001]()  
* My IPython notebook containing code of training and model: [Traffic_Sign_Classifier.ipynb](https://github.com/nutmas/CarND-LaneLines-P1/blob/master/CarND-LaneLines-P1/Traffic_Sign_Classifier.ipynb)  
* HTML version of IPython notebook: [Traffic_Sign_Classifier.html]()  
* Write-up of my project: [my_writeup_template.md](https://github.com/nutmas/CarND-LaneLines-P1/blob/master/CarND-LaneLines-P1/my_writeup_template.md).

### Dataset
To run this project fully, the dataset is required:

* [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

### Other files

* Sample signs gathered from internet and used for testing [donor_signs](./donor_signs)
* Outputs of the model train, validate and test process [output_data](./output_data)
* CSV file to show sign category and name [signnames.csv](./signnames.csv)
