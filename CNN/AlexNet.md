# ImageNet classification with deep convolutional neural networks

## Pass 1

    Title, Abstract, and conclusion. What the paper is talking about (15min around)

Researchers in this paper created a large, convolutional nural network which is used to classify images. The model they trained is considerably better than previous classification models. Besides, a new regularization method, dropout, is introduced to reduce overfitting in the fully-connected layers.

Reserachers believe the depth of the CNN model is important for achieving satisfactory classification results since removing any middle layer would lead to a 2 percent performance decrease in Top-1 classification.

Alex's team did not use unsupervised pre-training in this paper. Ultimately the research team plans to train this model based on video dataset since video's temporal structure provides information that is missing or far less obvious in static images.

## Pass 2

    From title to end. Omit those details such as proof. But need to knwo all figure and Table and their meaning in detail. May list important references if not familiar

Alex trained a neural network with 5 convolutional layers followed by 3 fully-connected layers, which would finally convert data into a 1000-way softmax. AlexNet is one of the largest convolutional neural networks. It contains a number of new and unusual features which improve its performance and reduce its training time. Several effective techniques are used to prevent overfitting. Lastly, since Alex's team uses a GTX 580 with only 3 GB of memory, the training time can be reduced, and the performance can be easily improved by using a more powerful GPU.

Alex's team chose the ILSVRC-2010 dataset since it's the only dataset for which the test set labels are available. They needed to down-sample the images to a fixed resolution of 256x256 before using them, as the dataset consists of variable-resolution images.

The architecture of AlexNet includes eight learned layers. AlexNet uses ReLU because it is several times faster than the equivalents with tanh units. The two CNN networks communicate with each other between layers 2-3 and the fully-connected layers.

Alex's team uses two primary ways to combat overfitting. The first is Data Augmentation, which is computationally free since they transformed images on the CPU. The next is Dropout.

Alex's team trained the model using SGD with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005. During this, they found that the small weight decay is important for training the model.

## Pass 3

    Know exactly the meaning of each word and sentence. Simulate whole training process in mind (2 hour around)
