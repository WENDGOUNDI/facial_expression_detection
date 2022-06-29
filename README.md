# facial_expression_detection
Facial expression has been approached in this porject. The dataset has 5 classes: angry, happy, neutral, sad, and surprise. The training set is large of 24282 images and 5937 images for the validation set. Little VGG-16 has been used for training.


#  Training Steps:
 - Import libraries
 - Set hyperparmeters and paths
 - Apply data augmentation to help the model to generalize well and increase the data size
 - Build a sequential model, the little vgg16
 - Apply ModelCheckpoint, EarlyStopping, ReduceLROnPlateau to have more control on the training
 - Train the model and display the training performance
 
 
# INFERENCE
To perform the inference, two face detectors have used: the haarcascade classifer and the mtcnn face detector then we apply the CNN trained model.
 - Inference 1: haarcascade classifer + weigths
 - Inference 2: mtcnn + weigths
 
 
![emotion_detected_2](https://user-images.githubusercontent.com/48753146/176390396-b34bc82c-0e2a-46bc-a599-e1a53c31b754.png)

# Training accuracy and loss
For this training, little VGG-16 has been used. We can observe that the training accuracy is not great. However, data augmentation has been used. To improve the model accuracy, we can trained for longer epochs and finetune the model. We can also apply some optimization techniques.

![loss_accuracy_2](https://user-images.githubusercontent.com/48753146/176390696-96563f26-549a-4d20-b63f-30d27a817eb4.png)
![loss_accuracy](https://user-images.githubusercontent.com/48753146/176390705-fcbda7d0-033e-4e56-a181-c723f1dc3d4c.png)


**Notes:**

- For improvement, train the network for longer epochs, apply different deep leraning techniques to improve the accuracy and try different face detector model during the inference.
- To run the python format for the inference file, it is advisable to make some adjustment. The file contains inference for image and video for both face detector models. Seperate them and run one at a time.
