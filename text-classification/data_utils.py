import sys
import os
import pickle

import numpy as np

"""
Utility functions for handling dataset, embeddings and batches
"""


def convert_file(filepath, word_dict, maxlen):
    dataset = []
    all_lines = open(filepath, 'r').readlines()
    for line in all_lines:
        line = line.strip()
        curr_set = [word_dict.get(w, 0) for w in line.split(' ')]
        if len(curr_set) > maxlen:
            dataset.append(curr_set[:maxlen])
        else:
            for i in range(maxlen-len(curr_set)):
                curr_set.append(0)
            dataset.append(curr_set)
    return dataset

def discover_dataset(path, wdict):
    dataset = []
    for root, _, files in os.walk(path):
        for sfile in [f for f in files if '.txt' in f]:
            filepath = os.path.join(root, sfile)
            dataset.append(convert_file(filepath, wdict))
    return dataset


def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen - len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen])
         for r in dataset])


# Class for dataset related operations
class EmotionDataset():
    def __init__(self, path, dict_path, maxlen=128):
        anger_path = os.path.join(path, 'anger/anger.txt')
        disgust_path = os.path.join(path, 'disgust/disgust.txt')
        fear_path = os.path.join(path, 'fear/fear.txt')
        joy_path = os.path.join(path, 'joy/joy.txt')
        sadness_path = os.path.join(path, 'sadness/sadness.txt')
        # TODO: all 5 datasets <DONE>

        with open(dict_path, 'rb') as dfile:
            wdict = pickle.load(dfile)

        self.anger_dataset = np.array(convert_file(anger_path, wdict, maxlen)).astype('i')
        print("anger", len(convert_file(anger_path, wdict, maxlen)))
        self.disgust_dataset = np.array(convert_file(disgust_path, wdict, maxlen)).astype('i')
        self.fear_dataset = np.array(convert_file(fear_path, wdict, maxlen)).astype('i')
        self.joy_dataset = np.array(convert_file(joy_path, wdict, maxlen)).astype('i')
        self.sadness_dataset = np.array(convert_file(sadness_path, wdict, maxlen)).astype('i')
        # TODO: all 5 datasets <DONE>

    def __len__(self):
        return len(self.anger_dataset) + len(self.disgust_dataset) + len(self.fear_dataset) + len(self.joy_dataset) + len(self.sadness_dataset)

    '''
    def get_example(self, i):
        is_neg = i >= len(self.pos_dataset)
        dataset = self.neg_dataset if is_neg else self.pos_dataset
        idx = i - len(self.pos_dataset) if is_neg else i
        label = [0, 1] if is_neg else [1, 0]

        print(type(dataset[idx]))
        return (dataset[idx], np.array(label, dtype=np.int32))

    def get_example(self, i):
        is_anger = i >= len(self.anger_dataset)
        dataset = self.anger_dataset if is_anger #else self.pos_dataset
        idx = i - len(self.anger_dataset) if is_anger else i
        label = [1, 0] if is_anger #else [1, 0]

        print(type(dataset[idx]))
        return (dataset[idx], np.array(label, dtype=np.int32))
        # TODO: all 5
    '''
    def load(self):

        dataset = np.concatenate((self.anger_dataset, self.disgust_dataset, self.fear_dataset, self.joy_dataset, self.sadness_dataset))
        labels = []

        for idx in range(0, len(self.anger_dataset)):
            labels.append([1, 0, 0, 0, 0])

        for idx in range(0, len(self.disgust_dataset)):
            labels.append([0, 1, 0, 0, 0])

        for idx in range(0, len(self.fear_dataset)):
            labels.append([0, 0, 1, 0, 0])

        for idx in range(0, len(self.joy_dataset)):
            labels.append([0, 0, 0, 1, 0])

        for idx in range(0, len(self.sadness_dataset)):
            labels.append([0, 0, 0, 0, 1])
        # TODO: all 5, check labels <DONE>
        return dataset, np.array(labels, dtype=np.int32)


# Function for handling word embeddings
def load_embeddings(path, size, dimensions):
    embedding_matrix = np.zeros((size, dimensions), dtype=np.float32)

    # As embedding matrix could be quite big we 'stream' it into output file
    # chunk by chunk. One chunk shape could be [size // 10, dimensions].
    # So to load whole matrix we read the file until it's exhausted.
    size = os.stat(path).st_size
    with open(path, 'rb') as ifile:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(ifile)
            chunk_size = chunk.shape[0]
            embedding_matrix[idx:idx + chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return embedding_matrix


# Function for creating batches
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    print("Generating batch iterator ...")
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

