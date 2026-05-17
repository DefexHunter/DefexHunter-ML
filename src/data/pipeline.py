import numpy as np
import pandas as pd

def load_data(path):
    dataset = pd.read_csv(path)
    print ("----------------Finished loading data----------------")
    print("Shape:", dataset.shape)
    return dataset


def clean_data(dataset):
    dataset['defects'] = dataset['defects'].astype('float')
    dataset.dropna(axis=0, inplace=True)
    print ("----------------Finished cleaning data----------------")
    print("Shape after cleaning:", dataset.shape)
    return dataset
