# **Traffic Sign Recognition: Writeup** 

___

### PROJECT SPECIFICATION

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarise and visualise the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyse the softmax probabilities of the new images
* Summarise the results with a written report

___


[//]: # (Image References)

[image1]: ./output_data/HistogramDataExplore.png "Category Histogram Visual"
[image2]: ./output_data/OriginalEachImagesample.png "Original Images Visual"
[image3]: ./output_data/NormalisedEachImagesample.png "GrayScale ImageSet"
[image4]: ./output_data/ModelAccuracy.png "Training Results"
[image5]: ./output_data/OriginalExtraSigns.png "Extra Signs Original"
[image6]: ./output_data/NormalisedExtraSignsPredictionResult.png "Extra Results"
[image7]: ./output_data/PreviousModelNormalisedExtraSignsPredictionResult.png "Previous Model Extra results"
[image8]: ./output_data/1NormalisedExtraSignsSoftmaxTop5Results.png "Top 5 1"
[image9]: ./output_data/2NormalisedExtraSignsSoftmaxTop5Results.png "Top 5 2"
[image10]: ./output_data/10NormalisedExtraSignsSoftmaxTop5Results.png "Top 5 10"
[image11]: ./output_data/11NormalisedExtraSignsSoftmaxTop5Results.png "Top 5 11"
[image12]: ./output_data/12NormalisedExtraSignsSoftmaxTop5Results.png "Top 5 12"
[image13]: ./output_data/14NormalisedExtraSignsSoftmaxTop5Results.png "Top 5 14"
[image14]: ./output_data/conv1_output_2.png "conv1"
[image15]: ./output_data/conv1_activation_output_2.png "conv1_relu"
[image16]: ./output_data/conv1_pooling_output_2.png "conv1_pool"
[image17]: ./output_data/conv2_output_2.png "conv2"
[image18]: ./output_data/conv2_activation_output_2.png "conv2_relu"
[image19]: ./output_data/conv2_pooling_output_2.png "conv2_pool"


Each of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) are addressed; below explains how I implemented them.

### Files Submitted

The project submission includes following required files:

* Ipython notebook with code [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb)
* HTML output of the code [Traffic_Sign_Classifier.html](./Traffic_Sign_Classifier.html)
* A writeup report [my_writeup_template.md](./my_writeup_template.md)

The link to the  git repository for this project is: [Traffic Sign Recognition](https://)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

The dataset I used for this project is the **([German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset))** and is available by clicking the link.

To understand some basic characteristics of the data set I used standard python function of 'len' applied to each variable the dataset had been assigned to. To Calculate the number of classes I applied a 'np.unique' function to the dataset variable that holds the labels:

``` 
n_classes = len(np.unique(y_train)) 
```

This is a numpy function which returns the sorted unique elements of an array.  
After performing these calculations on the traffic signs data set, the summary statistics can be understood:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is 32 pixels x 32 pixels x 3 channels (colour)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualisation of the dataset.

Understanding the sizes of each set a basic percentage split could be established but it is unclear if this split goes down to a category level.  
Exploring the dataset further I created a histogram of the 3 parts of the dataset and overlaid them as shown in the graph below:

![alt text][image1]

This histogram has 43 bins to match the 43 categories the dataset.  
This graph provides an easy way to visualise the percentage splits of training, validation and test data across the range of categories. X axis shows the category ID and the Y axis shows the total number of images for each category bin.  
I could conclude from this visualisation:  

* Some categories had far more images when compared with others.
* Percentage ratio of test:validation:training was approximately the same for each category.

Exploring the data even further:

+ Printed out a sample image for each category from the dataset;
+ Cross referenced the category number with the 'signnames.csv' to have an easy to understand title against each image;
+ Printed the exact count of each image for training:validation:test.
+ Identified the minimum and max colour value of a pixel in the image

These images are shown below:

![alt text][image2]

Viewing the dataset like this I was able to visualise and therefore familiarise myself with the data; It allowed an understanding of the variance in the set, the quality and gain an appreciation of the image recognition task that I was trying to perform; It immediately highlighted some challenges - images appeared very dark, blurred, pixelated, multiple signs - all which make it challenging for a human to recognise.

### Design and Test a Model Architecture

#### 1. Preprocessing

In order to get an understanding of the whole process I built a traffic sign classifier using tensorflow which simply consisted of one single fully connected layer. I adopted this to train on the training set and evaluated the results. They were extremely poor - 32% performance on the validation set.  
However training this model for many Epoch (2000 only took one hour) took very little time on a non-GPU computer. Using this lightweight classifier I was able to trial numerous methods of image pre-processing to understand what gives the best results for the neural net.
This experiment managed to find some pre-processing techniques that took the model from 32% to 66% with a low number of EPOCH.  
The preprocessing that resulted from this experiment was: 

* normalise using min/max scaling
    - Scale the range of pixel intensity values to reduce the range of distribution of feature values in the neural network when training
* Convert to Grayscale using numpy.mean function
    - simplify the image from a 3D matrix to a 1D matrix.
* Equalise the image with scikit exposure.equalize_hist function
    - take an image with low contrast and enhance it by spreading out the most frequent intensity values.

This array of images is the same 43 original image classifications that are shown in the visual exploration above, but with the preprocessing applied:

![alt text][image3]

When comparing the the two image results it is clear to see there is an improvement in the contrast making it possible to identify more features in the image. The range of intensities have a min and max of 0 and 1 respectively which will ease the amount of corrections the model might have to do to train. 

#### 2. Model Architecture

The model architecture I used was LeNet architecture with some modifications to add dropout after the first 2 fully connected layers. This model can be found in the python notebook in cell **[14]**

My final model consisted of the following layers:

| Layer                 |     Description                               | 
| ---------------------:|:--------------------------------------------- | 
| Input                 | 32x32x1 Grayscale image                       | 
| Convolution 3x3       | 1x1 stride, 1x1 padding, outputs 28x28x6      |
| RELU                  | Activation layer                              |
| Max pooling           | 2x2 stride,  outputs 14x14x6                  |
| Convolution 3x3       | 1x1 stride, 1x1 padding, outputs 10x10x16     |
| RELU                  | Activation layer                              |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | Output 1D shape                               |
| Fully connected       | 400 inputs, 120 outputs                       |
| RELU                  | Activation layer                              |
| Dropout               | Drop 50% of neurons to avoid over training    |
| Fully connected       | 120 inputs, 84 outputs                        |
| RELU                  | Activation layer                              |
| Dropout               | Drop 50% of neurons to avoid over training    |
| Fully connected       | 84 inputs, 43 outputs - 1 per sign label      |


#### 3. Model Training & Solution Approach

The final set of parameters used for the model were:

Training Rate: 0.00098
EPOCH: 75
Batch Size: 142
Dropout: 0.5

Final Training time: 51 mins 29 seconds (3089 seconds - actual time which may have included other tasks the computer sent to the processor during training time.)

The optimiser used was the adam optimiser. This optimiser is an extension to the standard Stochastic Gradient Descent(SGD) which updates the network weights based on training data. The difference with adam optimiser is instead of maintaining a single learning rate for all the weights and never changing during training as with SGD, it has an adaptive gradient algorithm which maintains a learning rate for each network weight, which can change during training.

The approximate process followed for training to reach these final set of parameters:

1. **Train LeNet model with preprocessed road sign data set.**
    * Default Training rate: 0.001; Batch size = 128; No Dropout No Shuffle.
    * With this basic untouched model I tried varying the EPOCH from 7 to 15. The results were quite poor and achieved 0.78 accuracy. After EPOCH 12 the model was over fitting as the training data was 100% but the verification was very low in the region of 35%.
2. **Added shuffle to each EPOCH cycle**.
    * At each EPOCH cycle the shuffle function will reorder the dataset to ensure the training data isn't biased by the order of the images.
    * shuffle is a function from 'sklearn.utils' and is implemented as shown:  
``` X_train, y_train = shuffle(X_train, y_train) ``` 
    * This produced some good results and the model training increased to 89% - 92.7%, but it was still limited to the same EPOCH range 7 - 12 and anything greater started over fitting - the 93% target could not be reached.
3. **Add Dropout after first and second fully connected layers**.
    * The dropout function drops/forgets a percentage of trained neurons at each EPOCH in order to force the model to find new weight and bias solutions to fit the training data.
    * Tensorflow has function called dropout..... it is implemented in the LeNet model architecture after fully connected layers one and two:  
``` fc1 = tf.nn.dropout(fc1, keep_prob, name='fc1_dropout') ```
    * Adding the dropout overcame the overfitting issue I was encountering and the 15 EPOCH were not over fitting - the 93% threshold was breached.
4. **Alter training rate and batch size**
    * The number of EPOCH could be increased and the validation result was 94.4% so I progressed to running the test set through the model. This produced some disappointing results - the training data came in with an accuracy of 88%.
    * To try better I decided to alter the batch size to understand its affect on the training. I tried simply doubling the size to 256, and ran the model. 
    * This modification produced worse results in verification with all other parameters remaining the same. I wasn't sure if my CPU could handle the batch size so reduced it down to 160. I could see the rate which the training and verification edge up was altered by this modification.
    * I adjusted the training rate down 2 points and settled on a batch size of 142. The model was ran with 65 EPOCH; this produced training accuracy 0.997, Validation 0.954, and a test accuracy of 0.931. This **was** my final model.
5. **Increase EPOCH to push model a little further**
    * I had an issue where I needed to train the model again; I could see there was still some spare capacity for training so edged up the EPOCH to see if I could get any better results. The EPOCH was increased to 75.
    * This was the final training loop and the final results are shown.
6. **Accuracy Results**
    * I ran the test data through the trained model and obtained the test  results.
    * Below are the accuracy results for all 3 areas of the test data.

**Training Accuracy = 0.999  Validation Accuracy = 0.968  Test Accuracy = 0.940** 

* The graph below shows the training data and validation data during the training. In the early stage ~ 7 EPOCH, it can be seen that the validation accuracy splits and drifts away from the steady training data set accuracy. However during the increase in EPOCH, the training data gradient is quite flat, but the validation data accuracy is converging towards the training accuracy level.
* With more EPOCH there maybe more room to increase the validation accuracy to higher than 96.8% but this would increase the CPU processing time.

![alt text][image4] 


### Test a Model on New Images

#### 1. Acquiring new images: I chose the following 15 signs to test as new images on my model.

Performing a simple internet search for German sign images I was able to obtain a selection of different signs which I could use for testing.

Below is an image of each of the signs with some summary information:

![alt text][image5]

The image set is a mix of challenges:  

* Images with no background
* Images containing multiple sign information
* A sign image taken from an illustration not a sign seen on the roadside
* Duplicate of the same image, but cropping preparation is different.

In order to prepare the images I started trying to manual crop them square and reduce the size to 32x32 to match the model input.
This was a lot of work and I replace this method with a simple square crop of the image then code to reduce the size required for the model input.

Using an OpenCV function the program resized the images while importing them:

```
resize = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA ) 
```

With the duplicated 'children crossing' sign; I cropped one image to a square size, and the other I left irregular shaped - when the resize function ran it 'squished' the non-square image creating a skew of the sign.

#### 2. Performance on New Images - Model's predictions on new traffic signs 

Here are the results of the prediction:

![alt text][image6]

Running the signs through the same evaluation code as the main datasets, the following result was output:

**Sign Recognition Accuracy = 60.00 %**

This result is much lower than the result the 94% accuracy the test data set showed.

As the new dataset is so small and the requirement was only 5 signs, it would have been easy to manipulate the result and select the signs that work and declare 100% match and success. However I wanted to understand the discrepancy:

A basic method to understand the result, I looked at the signs that failed and looked at the summary characteristics of the main dataset:

* 50kph - this has a very large dataset to train and test from, but this particular image was not from a real world sign, but an illustrated image.
* Children Crossing - both failed and even though both images were not same perspective they both produce the same prediction 'right of way at next crossing' - which could be a similar shape within the triangle area. The amount of data used for training of the children crossing was small. Although not the smallest quantity it was only 20% the amount of some of the signs and 50% of quantity of data available for 'right of way at next crossing'.
* 30kph - again this has a very large dataset, but this image from the internet has an additional sign beneath it and the perspective is looking from the ground up.
* Bicycle crossing - The training dataset for this sign was amongst the smallest quantities available. It classified it as 'turn left ahead' which is a completely different shape and colour sign, so this needs further investigation.
* End of all speed and passing limits - Again amongst the smallest quantity in the training dataset, but the sign is a very simple white circle with a black line through it. The pole either side of the sign may have caused the classifier some confusion.

I still had the output from the previous model which trained to a lower validation than this fial model and ran for only 65 EPOCH.

The output of the exact same sign images is shown below:

![alt text][image7]

This output was a worse performer than the final model, classifying 8/15 where as the final model gave 9/15 - which gives some confidence that the extra 10 EPOCH improved the model.

The signs classified were consistent between the two models for failures and success, however the previous model did manage to classify the skewed 'children crossing' road sign correctly.

#### 3. Model Certainty - Predictions considering the top 5 softmax probabilities.

To visualise the top 5 softmax probabilities I positioned the image of the internet sign and then a bar chart to represent the top 5 probabilities for each sign.

Where a positive classification is made the majority of the top 5 were 100% weighted on the top classification; All of the information can be seen in the python notebook cell **[30]** or the html export.

I will discuss the interesting signs which didn't classify:

![alt text][image8]
![alt text][image9]

As previously discussed the 'children crossing' classified as 'right of way at next crossing'. The top 5 results show that children crossing were a serious probability but the weighting shifted towards the right of way sign. An interesting area is the skewed sign was less weighted towards the right of way and children crossing and pedestrians were starting to get some attention. The previous model I had the data from managed to classify the skewed image correctly.  
**From this brief investigation, I would probably try to increase the amount of children crossing images, or at least equalise their quantity with the right of way to see if the classification improves**

![alt text][image10]

Looking at the top 5 for the 30kph classification, the 30kph does not even appear in the list. The signs it does list I can imagine having a small rectangular sign under them to provide some more information to support the sign.  
**At this stage a guess that the classifier is looking at the rectangular sign more to classify the sign. A more in depth analysis of the net would be required at a feature detection level to understand what is influencing the incorrect classification, before attempting to try to resolve it.**

![alt text][image11]

With the 'bicycle crossing' sign the correct classification is in the top 5 - albeit number 5, but it has no weighting. The top two with approximately equal weighting are both round signs with arrows.  
**This is very difficult to comprehend why the net swayed towards the round signs to classify this image, increasing the amount of training data for the bicycles crossing could possibly improve the situation**

![alt text][image12]

Speed limit 50kph - looking at the top 5 the top 2 are 30kph or 50kph. In this classification the model weighted everything to the 30kph. Both of the data sets have a similar fairly large amount of training data.  
**Again an investigation in the net features might reveal it had the bottom half of the 5 as a feature which matched the 3, and the top half of the 5 in this image was blurred or even too perfect for it to think it was a 5**

![alt text][image13]

End of all speed and passing limits - looking at the probabilities it was second in the top 5 with a confidence level of ~40% opposed to the number one spot of 45%.  
**As in the previous discussion it is a simple sign and quite a good contrast so my expectation that this should be easier to classify than a lot of the other signs, which points me towards there just wasn't enough training data and this should be increased to equalise it's chances of being weighted correctly during training.**

### Visualising the Neural Network 
#### 1. Visualise the trained network's feature maps.

I carried out the visualisation of a number of signs to understand what the feature map looked like and how the net went about classifying a sign.
I pulled off the data for the initial convoluted layers 1 and 2 including the ReLU and pooling.

With the first convolutional layer it was possible to make out some off the the shapes the model was extracting to identify it, but once it entered the second convolution the data became incredibly difficult to understand as i think the model was breaking the basic features into smaller unique arrangement of pixels to uniquely identify the sign.

The output of the skewed 'children crossing' was one sign of interest I tried to look at the data to understand what the classifier was doing:

**1st Convolutional output**  
![alt text][image14]

**1st Convolutional ReLU**  
![alt text][image15]

**1st Convolutional pooling**  
![alt text][image16]

**2nd Convolutional output**  
![alt text][image17]

**2nd Convolutional ReLU**  
![alt text][image18]

**2nd Convolutional pooling**  
![alt text][image19]

As can be seen in the earlier stages the triangular form of the sign, and the two shadow images of people were focused on in the feature maps.
On the second convolution the lies of the triangle can still be vaguely seen but beyond that it became an array of pixels. 
**In feature maps 3, 7 and 11 of the conv2 ReLU the straight edges of the triangle can be still seen as the feature.**
