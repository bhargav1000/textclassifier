# EMOTION-DETECTOR
A Convolutional Neural Network (CNN) based text classifier used to detect emotions in a given utterance using Python. 
The ISEAR dataset was used to train a CNN to detect 5 types of emotions: Joy, Fear, Anger, Sadness and Disgust. [GloVe](https://nlp.stanford.edu/projects/glove/) vectors were used to create embeddings to be used in the CNN layers to detect features commonly found in a given emotion. The model has a validation accuracy of 64%.

## Getting Started:
- Download the [GloVe dataset](http://nlp.stanford.edu/data/glove.42B.300d.zip).

- Generate embeddings using ```python3 embeddings.py -d ./data/glove.42B.300d.txt --npy_output ./data/newembeddings.npy --dict_output ./data/newvocab.pckl --dict_whitelist ./data/sorted.vocab```

- Data sorted folders are as follows: ```/text-classification/data``` = training dataset | ```/text-classification/test``` = test dataset

- Run ```/text-classification/newtrain.py``` to train the model. 

## Reasons why this model failed:
1. **Thin dataset.** The dataset used had only 1300 instances per class and showed high variations in the sentences in each instance of the class. Model showed consistent signs of overfit even after cross validation. 

2. **Computationally Ineffective model.** CNN is not the right model for emotion detection because emotions are usually characterized by visible patterns in the given sentence. For effective classification, each sentence of the given dataset must be broken down into sequences and these sequences must be memorized to provide patterns consistent with a given emotion. For this purpose a **LSTM(Long Short Term Memory) Network** can be used to identify the sequences used in a given sentence and associate the patterns to a given emotion class.

3. **Class Imbalance.** The dataset cannot provide full or exhaustive sequences to classify emotions and their polarity in a given sentence. Also GloVe vectors are not sufficient to match with the generated sequences in a CNN model because of small variations of certain words in a given sentence.


## TODO:
- Use [ELMo vectors](https://allennlp.org/elmo) to account for typos and other variations in words for better classification.
- Perform some more text pre-processing steps such as Stemming and also look for repetitive patterns.
- Use LSTM models. 
