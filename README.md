# Dog-Breed-Classification


I had taken part in kaggle comepition which was a playground competition for dog breed classification. The notebook describes about how to efficently predict and classify different breeds of the dog.  Due to very huge dataset and because of our computational limitation I used google colab to get the GPU power on colab. Using labels of each images the first task was to preprocess the dataset which included to encode the multiclass labels into numeric values using OneHotEncoding and then load the train and test image files and convert them into NumPy arrays. There were total 10,222 training images and 10,300 test images. Second task was to make all the input images of same size in order to do that I used Keras Vgg16 pre-process input. Convolutional Neural Network(CNN) was the base algorithm for the classification. I had used ImageNet pre-trained weights in order to train my model. Different models were implemented on bases of trial and error using Keras application models. My first model used Vgg16 model which took (224,224,3) as image input size and trained over 120 classes. I then used ResNet50 over Vgg16 model in order to make the model more tunned and to overcome the overfitting. Other models such as Inception, Xception, ResNet101 and also developed ensemble model containing a mix of three models(Inception, Xception, ResNet50) were used. In one of my models I used ResNet50 pretrained model and extracted the bottleneck weights of ImageNet dataset and run a logistic regression and a SVM classifier to see the performance. Regularizers, various pooling techniques like GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout was also incorporated in order to tune the model and lead the competition.


## Exploratory data analysis and Preprocessing

* Converting images into numpy arrays
* Encoding text labels

## Deep learning methodology

I have used following deep neural networks to classify and compare performance against each other.

* Simple CNN with 5 layers and softmax activation
* VGG-16
* VGG-19
* ResNet-51
* ResNet-101
* Inception Resnet


## Evaluation Metric and loss

* Accuracy
* Categorical Crossentropy loss



## Requirements

* keras
* tensorflow
* numpy


