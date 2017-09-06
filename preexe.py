import pandas as pd
import pickle
import random
import time

from sklearn.model_selection import train_test_split

def get_data():
    pass

def split_data():
    df = pd.read_csv("train.csv")
    columns = ['question1', 'question2', 'is_duplicate']
    df = df[columns].fillna(value="a")

    data = df[['question1', 'question2']].values
    label = df[['is_duplicate']].values

    x_train, x_test, y_train, y_test =  train_test_split(data, label, test_size=0.2)
    pickle.dump([x_train, x_test, y_train, y_test], open("data.pkl", "wb"))
    return x_train, x_test, y_train, y_test

def get_batch(batch_size):
    x_train, x_test, y_train, y_test = pickle.load(open("data.pkl", "rb"))
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
            print (sentence[0])
            length = len(sentence[0])
        if isinstance(sentence[1], str) and len(sentence[1]) > length:
            print (sentence[1])
            length = len(sentence[1])

        if isinstance(sentence[0], str): all_length.add(len(sentence[0]))
        if isinstance(sentence[1], str): all_length.add(len(sentence[1]))
    print (all_length)
    return length

if __name__ == "__main__":
    print (Max_document_length())
    