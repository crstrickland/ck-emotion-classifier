# Cohn-Kanade Emotion Classifier

This is a classifier using the pretrained weights of the [OpenFace](https://cmusatyalab.github.io/openface/) nn4.small2 DNN model with a dense classification layer. The final classification layer is trained on a set of labeled faces from the [Cohn-Kanade AU-Coded Expression Database](http://www.pitt.edu/~emotion/ck-spread.htm).

The CK database only has emotion labels for 327 of the images included. Fortunately, these images are the last in a sequence of a subject's emotion expression. Here's a pair of images after cropping and alignment: 

![Image 11 of the sequence](/emotions/S044_003_00000011.png?raw=true) ![Image 14 of the sequence](/emotions/S044_003_00000014.png?raw=true)

Images 11 and 14 of this sequence (14 is the final element) are both very good examples of happiness. Taking the last four images of each labeled sequence and partitioning them into training and validation sets gives 931 training and 381 validation examples. The images are further resized to 96x96 using Keras, and after training the model yields a respectable 84% accuracy on the test data (the 5th image from each sequence). 

![confusion matrix of test data](/emotions/cm_best.png?raw=true "Confusion matrix of best model checkpoint")

Even with small datasets, transfer learning can give great results!
