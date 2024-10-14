# Comparative Study of LLMs for Text Classification and QA
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

Since it worked nicely, the same approach as before was used for this task.

Instead of the GPT Model, for this task BERT from HuggingFace was used.

Now it is enough to run `3_2.py` that includes both the preprocessing and the training part.

### Dataset preprocessing

Each entry in the dataset is composed of 6 text fields:
- the context (called "article")
- the question
- four answers

The idea was to train a model capable of selecting the correct answer within the four.
We decided to represent each answer with a vector as before that also cotains information from the context and the question.

To create the embeddings of the four answers we followed these steps:
1. using BERT we retrieved the [CLS] token's embedding of the context, the question and the answers
2. each answer is then represented by the average of its embedding and context's and article's embeddings

The four embeddings are then put in a tensor that represents the whole dataset in this way: first the correct one and then the other three.

Each entry of Race is processed to create the train, the validation and the test dataset.


### Training and evaluation

The learning process was done by selecting Margin Ranking Loss as a metric provided bt pyTorch.
For each question the following was done:
1. the score for each of the 4 responses is calculated, providing as input the vector representing them;
2. the loss is calculated 3 times, comparing the score of the correct answer with that of the three incorrect ones.

The idea is to push the model to predict higher correct answer scores while keeping incorrect answer scores low.

To penalize similar scores, the `margin` parameter was set equal to 1 (arbitrary choice not based on particular information).
The results obtained by training the model for 100 epochs are here reported:

![image](https://github.com/simogiovannini/DLA-lab2/assets/53260220/1e57861e-325d-49dd-9128-aa30269863cf)

![image](https://github.com/simogiovannini/DLA-lab2/assets/53260220/255068d9-435d-4ffd-9f79-3d4d35f53210)

![image](https://github.com/simogiovannini/DLA-lab2/assets/53260220/b000979d-38ee-48b1-9822-97e3d0510a08)

it can be seen that the technique used does not work very well. around 250 epochs we begin to detect a marked overfitting phenomenon as the loss calculated on the validation set begins to increase while the training loss continues to decrease.

The maximum accuracy obtained is about 38 percent, so you have a model that is not able to be used effectively.

In order to make another attempt, a new loss function implemented on the basis of intuition was tried.

Given the estimated scores for the four answers the loss is calculated as the sum of the three ratios of the score of the correct answer to each of the score of the wrong answers.

The idea was to push the model more to create more discrepancy between the correct answer and the other three.

A run was run with this new loss (in pink) and an additional one going to increase the learning rate (in orange), as the loss had a significantly higher magnitude than the one previously used.

The results obtained are now shown.

![image](https://github.com/simogiovannini/DLA-lab2/assets/53260220/55c3bcf7-c0ea-4dc7-aeaf-124cee44e527)

![image](https://github.com/simogiovannini/DLA-lab2/assets/53260220/9421096c-1118-487e-9534-70e20c740b12)

it can be seen that the attempt was more unsuccessful than the previous one in that, at 500 epochs, only a very slight decrease in the loss function is noted and that the accuracy still stands at 26 percent, which is totally equivalent to choosing an answer randomly. The model trained in this way is completely ineffective.

## Extra: training a GPT Model on 'La Divina Commedia'

By slightly modifying the code in [this repository](https://github.com/karpathy/nanoGPT/blob/master), a small version of GPT was trained using the text of Dante's Inferno from the Divine Comedy.

The code was launched by setting the following values for the parameters determining the architecture:

- `max_iters = 5000`
- `learning_rate = 3e-4`
- `n_head = 6`
- `n_layer = 6`

The said model thus has about 10 million parameters, and the training and validation loss trend in addition to text generated through the trained model are shown below.

![image](https://github.com/simogiovannini/DLA-lab2/assets/53260220/6efc86c7-3dc5-4586-8d09-b1e04e9b618e)

![image](https://github.com/simogiovannini/DLA-lab2/assets/53260220/5ddca491-b866-4412-b758-e3a4f0526ffb)

Given the size of the model and the size of the dataset, after 1500 epochs the model begins to overfit.

The text generated first from a model trained for 5000 epochs and then one trained for 1500 is then shown.

`piuavvien si che la faccia booru.`

`Taccia Lucria e Ciato Ii con terra che tanta di si capa, andi se' e ne costl pianto manto;`

`e vidi gente per lo capo sotto il viso ch'ebbe aver di si' trapassando ha fatta.`

`E come quella selva runa e 'l ti basso di sotto quel piuche 'n su resto, e ha piuche cocia gia l'univiso le belle a quai le picte sestagne.`

`Le quelle co' il braccio era e 'l petto, e di trieversi' di guale pietra che la pioggia, annima in si sedea quel che so'`

And:

`vegne luol che tene avea di lento per le tre tristo io dolce sangui.`

`E verranno a quello in ciel cerchi che, centro da l'imagine due fiso sanzar carnel non la' testa golte.`

`E che di lungi pieta, guarda elordo lo suo dicentro per lascionche; e s'io nodi piu felici mai.`

`La vendette tu sara conrona, ch'io di trei era simistri, primi era credaro stravi.`

`L'un ho campito inno er' campagno che quel che faccia e avoltte la zingua voglia de l'arte torna, or vede speranno;`

`vidi`

The second model seems better because it contains slightly more full words but in both cases the generated text is not understandable despite being in the style of the text used for learning


## Requirements
You can use the `requirements.txt` file to create the conda environment to run the code in this repository.
