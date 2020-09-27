from transformers import  BertTokenizer
import pandas as pd

def preprocess_data(dataset_path):

    data = pd.read_csv(dataset_path)

    # remove nans, if any
    data = data.dropna()

    # sample percentage of dataset
    data = data.sample(frac=0.1)

    # merge message, subject, relation, object as indicated
    data["Message"] = data["Message"] + " [SEP] " + data["Relation"] + " [SEP] " + data["Object"]
    data.drop(["Subject", "Relation", "Object", "Unnamed: 0"], axis=1, inplace=True)

    return data


