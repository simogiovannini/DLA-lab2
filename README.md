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


## Utils
In the utils directory there are two classes named "wrappers", we use them later only to access easily to the items in the datasets downloaded from HuggingFace.


## Exercise 1.1: Text Classifier

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


## Exercise 2.1: Training a text classifier
Before executing `3_1.py` the user must run `utils/dataset_synthetizer` to create the custom dataset to be used for this task.

### Dataset synthetization
In this script we downlaod the dataset from HuggingFace and then we process it through the GPT Model trained by HuggingFace.

Each sentence in the dataset is passed to the GPT Tokenizer and then the result is passed to the GPT Model to retrieve embeddings for each token.
The embedding of the whole sentence is the average of the embeddings of the tokens within it.

At the end of the iteration each sentence is represented by a tensor or 768 float. All the couples of vectors and labels are stored in a tensor that is saved in the datasets directory.

The script must be run two times: one for the train and one for the test split of the ag_news dataset.

### Training the model


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
