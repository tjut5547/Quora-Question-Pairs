import pandas as pd
import pickle
import random
import time, datetime
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn

def get_one_hot():
    x_train, x_test, y_train, y_test = pickle.load(open("data.pkl", "rb"))
    x_text = np.append(x_train, x_test).reshape(2 * len(list(x_train) + list(x_test)))

    vocab_processor = learn.preprocessing.VocabularyProcessor(40, min_frequency = 10)
    vocab_processor = vocab_processor.fit(x_text)
    vec_train = list(vocab_processor.transform(x_train.reshape(2 * len(x_train))))
    vec_test = list(vocab_processor.transform(x_test.reshape(2 * len(x_test))))
    vec_train = [[list(vec_train[index]), list(vec_train[2 * index + 1])] for index in range(int(len(vec_train) / 2))]
    vec_test = [[list(vec_test[index]), list(vec_test[2 * index + 1])] for index in range(int(len(vec_test) / 2))]
    print ("******************************************************************************************************")
    print (len(vocab_processor.vocabulary_))
    pickle.dump([vec_train, vec_test, y_train, x_train], open("one_hot.pkl", "wb"))
    print ("******************************************************************************************************")
    return vec_train, vec_test, y_train, x_train


def testing():
    df = pd.read_csv("train.csv")
    print (df.isnull().any())

def split_data():
    df = pd.read_csv("train.csv")
    columns = ['question1', 'question2', 'is_duplicate']
    df = df[columns].fillna(value="a")
    print (df.isnull().any())

    data = df[['question1', 'question2']].values
    data_a = df['question1'].values
    data_b = df['question2'].values
    label = df[['is_duplicate']].values

    x_train, x_test, y_train, y_test =  train_test_split(data, label, test_size=0.2)
    pickle.dump([x_train, x_test, y_train, y_test], open("data.pkl", "wb"))
    return x_train, x_test, y_train, y_test

def get_batch(batch_size):
    # x_train, x_test, y_train, y_test = get_one_hot()
    print (datetime.datetime.now().isoformat())
    x_train, x_test, y_train, y_test = pickle.load(open("one_hot.pkl", "rb"))
    data = list(zip(x_train, y_train))
    random.shuffle(data)

    for batch in range(0, len(data), batch_size):
        if batch + batch_size >= len(data):
            yield data[batch : len(data)]
        else:
            yield data[batch : (batch + batch_size)]

def Max_document_length():
    length = -1
    all_length = set()
    x_train, x_test, y_train, y_test = pickle.load(open("data.pkl", "rb"))
    for sentence in x_train:
        if isinstance(sentence[0], str) and len(sentence[0]) > length:
            length = len(sentence[0])
        if isinstance(sentence[1], str) and len(sentence[1]) > length:
            length = len(sentence[1])

        if isinstance(sentence[0], str): all_length.add(len(sentence[0]))
        if isinstance(sentence[1], str): all_length.add(len(sentence[1]))
    return length

if __name__ == "__main__":
    split_data()
    get_one_hot()
    