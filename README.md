# DLA-lab2  (README work in progress)
Deep Learning Applications laboratory on LLMs

## Models
In `models/` we define the three architectures that were tested during this laboratory.

### Text Classifier
A simple 4-layers MultiLayerPerceptron is implemented in `models/text_classifier.py`. The number of outputs is equal to the number of classes.

### QA Answering
The architecture of the Ranking Predictor used for the second exercise is defined in `models/ranking_predictor.py` and it's almost the same as the Text Predictor except for the size of the output which is always 1.

## Datasets

Links to the datasets used for the following exercises.

### Text Classifier

[Ag_news](https://huggingface.co/datasets/ag_news) was used.


### QA Answering

[Race](https://huggingface.co/datasets/race) was used.


## Exercise 1.1: Comparing Architectures

`1_1.py` contains a comparison between four architectures on CIFAR10:
- MultiLayerPerceptron
- ResNet34
- ConvNet34 (the same architecture as ResNet34 but without residual connections)
- ResNet50

Each model was trained for the same number of epochs, using the same train/validation split. Here we analyze the results:

![image](https://github.com/simogiovannini/DLA-lab1/assets/53260220/18ba47e7-1d1c-4084-866e-67e4a9c246fd)

![image](https://github.com/simogiovannini/DLA-lab1/assets/53260220/487c0372-9911-42ab-b0c2-47ab0ba28ee2)

MLP is completely uneffective on this task while all the CNNs share the same behaviour: they work nicely but the rapidly overfit. We can see it from the growth of validation loss over time.

There is no particular difference between ResNet34 and ResNet50.

At the end of the training each model was saved in `trained_models/`.


## Exercise 2.1: ResNet34 vs ConvNet34

From the last comparison it's clear how ResNet34 offers better performance than ConvNet34 and, thanks to our implementation, the only difference between them is the presence or not of skip connections.

![image](https://github.com/simogiovannini/DLA-lab1/assets/53260220/eba0393b-79ad-45cd-8d17-a38b76653135)

From the previous graph we can see that ResNet34 reaches lower values of loss in a lower number of epochs. So we tried to explain these phenomenon.

The first idea was to verify if that happens systematically or whether this difference had been the result of randomness due to a particular split in the data.

We then trained the two models for 20 epochs 10 times and collected information on all training sessions.

After this initial verification, it is possible to say that the skip connections lead to an actual improvement in performance.


## Exercise 2.3: Class Activation Map

We implemented Class Activation Map to analyze and explain the predictions of our implementation of CNN. The script loads the model saved by the previous scripts to clasify 16 images random sampled from the test test. The script saves a  `CAMs/pre_image_grid.png` that contains these 16 images.

For each image in the grid, we predict its class using the trained net and then we save an image that contains the class activation maps. Each CAM is saved in the same directory of the grid with the following file name: `CAMs/{ID}-{predicted_class}.png`.

Here we have some examples.

![pre_image_grid](https://github.com/simogiovannini/DLA-lab1/assets/53260220/bda6c940-6326-4394-8a1e-e88d6fe94105)

![1-bird](https://github.com/simogiovannini/DLA-lab1/assets/53260220/242edd81-f480-4088-8c1f-1abb0969a6e5)

![14-airplane](https://github.com/simogiovannini/DLA-lab1/assets/53260220/25f1a479-c59a-4fe6-883f-6db5c67e53c5)

![15-cat](https://github.com/simogiovannini/DLA-lab1/assets/53260220/700e586d-1425-47cb-9c19-76777b0cf5f8)


## Requirements
You can use the `requirements.txt` file to create the conda environment to run the code in this repository.
