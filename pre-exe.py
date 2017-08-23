import pandas as pd

def get_data():
    df = pd.read_csv("train.csv")
    columns = ['question1', 'question2', 'is_duplicate']
    df = df[columns]
    return df


if __name__ == "__main__":
    get_data()