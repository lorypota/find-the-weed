# find-the-weed
Convolutional Neural Network that detects weed and its direction (from the center of the image) based on circular images of 128px.

## Requirements
The required dependencies for this project are listed in the [requirements.txt](https://github.com/lorypota/find-the-weed/blob/main/requirements.txt) file.

## Dataset
Characteristics:

* **Composition:**  The dataset consists of 640x640 pixel images featuring various types of weeds.
* **Annotations:** Each image is annotated using the COCO format with bounding boxes (x, y, width, height), outlining the location of individual weeds within the images.

**Annotated full image (640x640 pixels):**

   ![Full Annotated Image](https://raw.githubusercontent.com/lorypota/find-the-weed/main/plot_full_image.png)

**Grid of annotated sub-images (128x128 pixels):**

   ![Grid of Sub-Images](https://raw.githubusercontent.com/lorypota/find-the-weed/main/plot_sub_images.png)

**Example of a sub-image (128x128 pixels):**

   ![Example Sub-Image](https://raw.githubusercontent.com/lorypota/find-the-weed/main/plot_sub_image.png)

The file weed_dataset_transform.py defines a PyTorch dataset. 
This dataset collects a picture of 128x128 pixels by randomly cropping inside the fully annotated image. 
Once the cropping is done, a circular mask is applied to the cropped image, along with a random rotation of the image and the annotations. This allows for an improved model precision at the end of the training process.
This same dataset is also used to extract the images that are fed into the final trained model.
By cropping a 128x128 image in a random position we allow for more test cases.

## Model
### Architecture
The model is based on a Faster R-CNN architecture (from torchvision) with the FastRCNNPredictor classifier.

### Transfer Learning
We decided to train a network through transfer learning using a pre-trained network called Faster R-CNN (used to recognize object position, as described in https://arxiv.org/abs/1506.01497). 
By using this pre-trained network we were able to accelerate our training process and improve the model's performance.

## Training Process
To train the network, we had to adapt to the input format required by the Faster R-CNN network, which consists of an image represented as a tensor of size (C, W, H) and a target represented as a dictionary containing "boxes" (the bounding boxes) indicating the position of the weeds and "labels" indicating which category each bounding box belongs to (in our case, weeds).

During training, we randomly crop the training images into an image of 128x128 pixels.
If this cropped image contains an annotation, it is used to train the model.
If not, it tries to crop again until it finds an image containing an annotation.
If after 100 tries none of the cropped images had an annotation, we force the centering of the cropping around the annotation contained inside the image.

The cropping was necessary because the network would not have been able to process 640x640 base size images, as the bounding boxes would have been too small compared to the image size, causing performance degradation issues.

## Evaluation process
The evaluation process has been carried out on the 5 trained models.
This process aims to find the best model in terms of precision and recall. To do so, it is important to study the quantity of true-positives, false-positives and false-negatives predicted by each model.
In order to calculate these parameters,the models are evaluated on the same set of randomly selected cropped images.

![Models evaluation](https://raw.githubusercontent.com/lorypota/find-the-weed/main/models_evaluation.png)

The model n.2 shows the highest precision but with one of the lowest recall.
The model n.4 is the most balanced in terms of precision and recall, therefore it is likely to be the most useful in most cases.

During evaluation, the models exhibited low recall. This is likely due to random image cropping during training, potentially leading to incomplete data and imprecise annotations. Cropping may have obscured relevant image features.

## Examples of classified data

* The red boxes are the bounding boxes defined inside of the annotations.
* The blue boxes are the predictions made by the model.
* The arrows indicate the direction in which the found boxes are, starting from the center of the image.

Note that these are only some examples and the model can vary in accuracy.

**Example 1:**

   ![Example 1](https://raw.githubusercontent.com/lorypota/find-the-weed/main/example1.png)

**Example 2:**

   ![Example 2](https://raw.githubusercontent.com/lorypota/find-the-weed/main/example2.png)

**Example 3:**

   ![Example 3](https://raw.githubusercontent.com/lorypota/find-the-weed/main/example3.png)
