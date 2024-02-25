# find-the-weed
Convolutional Neural Network that detects weed and its direction (from the center of the image) based on circular images of 128px.

## Dataset
Characteristics:

* **Composition:**  The dataset consists of 640x640 pixel images featuring various types of weeds. To ensure robustness, the images include variations in lighting conditions. 
* **Annotations:** Each image is annotated using the COCO format with bounding boxes (x, y, width, height), precisely outlining the location of individual weeds within the images.

In order to effectively train the model we decided to split each image of 640x640 pixels into 25 sub-images of 128x128 pixels. 

**Full Annotated Image (640x640 pixels):**

   ![Full Annotated Image](https://raw.githubusercontent.com/lorypota/find-the-weed/main/plot_full_image.png)

**Grid of Sub-images (128x128 pixels):**

   ![Grid of Sub-Images](https://raw.githubusercontent.com/lorypota/find-the-weed/main/plot_sub_images.png)

**Example Sub-image (128x128 pixels):**

   ![Example Sub-Image](https://raw.githubusercontent.com/lorypota/find-the-weed/main/plot_sub_image.png)

The file weed_dataset.py defines the PyTorch dataset utilized for the training process. 

The file weed_dataset_visual_and_direction.py defines a different PyTorch dataset without the subdivision of the images, this comes in handy during the final process of identifying weed and its direction. By cropping a 128x128 image in a random position inside of this bigger image we can pass it to the model and see how it evaluates it, allowing for more test cases.

## Model
### Architecture
* Base Network: The model is based on a ResNet-34 architecture (from torchvision) with the final classification layer removed.
* Modifications:
  * Channel Adjustment: A convolutional layer (Conv2d) is added to adjust the number of channels for compatibility with the subsequent classifier; 
  * Classifier: A custom classifier consisting of a flattening layer, followed by two fully connected linear layers (with ReLU activation in between), and a final linear output layer is used to predict the weed classes.

### Transfer Learning
We decided to train a network through transfer learning using a pre-trained network called FAST R-CNN (used to recognize object position, as described in https://arxiv.org/abs/1506.01497). 
By using this pre-trained network we were able to accelerate our training process and improve the model's performance.

## Training Process
To train the network, we had to adapt to the input format required by the FAST R-CNN network, which consists of an image represented as a tensor of size (C, W, H) and a target represented as a dictionary containing "boxes" (the bounding boxes) indicating the position of the weeds and "labels" indicating which category each bounding box belongs to (in our case, weeds).

During training, we split the training images one by one and divided them into 128x128 size images. We then trained the network using only the images containing weeds, along with their associated bounding boxes and labels (with label = 1, representing the weed category). This was necessary because the network would not have been able to process 640x640 base size images, as the bounding boxes would have been too small compared to the image size, causing performance degradation issues.

## Evaluation
TODO

## Examples of classified data
The red boxes are the boxes where the model detected some weed.
The arrows indicate the direction in which the found boxes are, starting from the center of the image.
The blue boxes are the correct annotations found inside the dataset.
Note that this are only some examples and the model can vary in accuracy.
The image is cropped circularly 
![Example 1]()
![Example 1]()
![Example 1]()

## Things to do
* finish evaluation process
* fix low quality when classifying (image seems to be too zoomed in)