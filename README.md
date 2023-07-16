# DLA-lab2
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


## Exercise 2.1: Training a text classifier
Before executing `3_1.py` the user must run `utils/dataset_synthetizer` to create the custom dataset to be used for this task.

### Dataset synthetization
In this script we downlaod the dataset from HuggingFace and then we process it through the GPT Model trained by HuggingFace.

Each sentence in the dataset is passed to the GPT Tokenizer and then the result is passed to the GPT Model to retrieve embeddings for each token.
The embedding of the whole sentence is the average of the embeddings of the tokens within it.

At the end of the iteration each sentence is represented by a tensor or 768 float. All the couples of vectors and labels are stored in a tensor that is saved in the datasets directory.

The script must be run two times: one for the train and one for the test split of the ag_news dataset.

### Training the model

Now the user can run `3_1.py`.

The two tensors created before are now used as training and testing datasets.

The text classifier is trained to classify these 768-sized vectors to their correct class of beloning.

![image](https://github.com/simogiovannini/DLA-lab2/assets/53260220/3633b782-fdcc-4d90-9fd2-1439877c9a57)

As shown above, results are satisying. In 100 epochs of training we reached an accuracy of around 87% with apparently no overfitting, according to what the validation loss chart indicates.


## Exercise 3.2: Training a multiple choice question answering model




## Requirements
You can use the `requirements.txt` file to create the conda environment to run the code in this repository.
